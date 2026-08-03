import sys
import io
import json
import os
import tempfile
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch
from contextlib import redirect_stdout

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from selfevolve_drive.critics import RuleBasedCritic
from selfevolve_drive.base_models import audit_opendrivevla, create_base_model, parse_opendrivevla_trajectory
from selfevolve_drive.pipeline import build_records, load_config
from selfevolve_drive.planner import Policy
from selfevolve_drive.reflection import reflect
from selfevolve_drive.simulator import generate_scenarios
from selfevolve_drive.self_evolution import run_self_evolution
from selfevolve_drive.vla_training_data import trajectory_text
from selfevolve_drive.web_demo import DATA_STORE, Handler, compare
from selfevolve_drive.demo_runtime import RuntimeEventLog


class CoreTests(unittest.TestCase):
    def test_deterministic_generation(self):
        self.assertEqual(generate_scenarios(3, 7), generate_scenarios(3, 7))

    def test_red_light_failure_is_reflected(self):
        s = generate_scenarios(1, 3)[0]
        s.traffic_light, s.stopline_distance, s.ego_speed, s.speed_limit = "red", 6.0, 14.0, 14.0
        t = Policy("baseline", seed=1).plan(s)
        c = RuleBasedCritic({"safety": .45, "rule": .35, "comfort": .20}).evaluate(s, t)
        self.assertIn("red_light_risk", c.failures)
        self.assertEqual(reflect(s, t, c).verdict, "revise")

    def test_pipeline_schema_and_size(self):
        cfg = load_config(ROOT / "configs" / "default.json"); cfg["num_samples"] = 40
        records, reward = build_records(cfg)
        self.assertEqual(len(records), 40)
        self.assertTrue({"scenario", "trajectory", "critic", "reflection", "quality"} <= records[0].keys())
        self.assertEqual(reward.coef.shape[1], 3)

    def test_multi_round_self_evolution_produces_memory(self):
        cfg = load_config(ROOT / "configs" / "default.json"); cfg["num_samples"] = 120
        records, _ = build_records(cfg)
        policies, history, memory = run_self_evolution(records, cfg["critic_weights"], rounds=2)
        self.assertEqual(len(policies), 2)
        self.assertEqual(len(history), 2)
        self.assertGreater(len(memory), 0)
        self.assertEqual(history[-1]["memory_size"], len(memory))

    def test_local_opendrivevla_assets_are_detected(self):
        audit = audit_opendrivevla(ROOT)
        self.assertTrue(audit["checkpoint_installed"])
        self.assertTrue(audit["code_complete"])
        self.assertEqual(audit["architecture"], "LlavaQwenForCausalLM")

    def test_lightweight_base_model_disclosure(self):
        meta = create_base_model(ROOT).metadata()
        self.assertEqual(meta["recommended_base_model"], "OpenDriveVLA-0.5B")
        self.assertTrue(meta["checkpoint_installed"])
        self.assertFalse(meta["weights_loaded"])

    def test_official_trajectory_parser(self):
        text = "<traj_start>[(1.0,0.0),(2.0,0.1),(3.0,0.2),(4.0,0.2),(5.0,0.1),(6.0,0.0)]<traj_end>"
        trajectory = parse_opendrivevla_trajectory(text)
        self.assertEqual(len(trajectory.points), 6)
        self.assertEqual(trajectory.points[-1][:2], [6.0, 0.0])
        self.assertGreater(trajectory.target_speed, 0)
        self.assertIn("<traj_start>", trajectory_text(trajectory))

    def test_cache_is_used_only_for_matching_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "cache.jsonl"
            cache.write_text(json.dumps({
                "scene_id": "cached_scene",
                "trajectory": {
                    "points": [[float(i), 0.0, 2.0] for i in range(1, 7)],
                    "target_speed": 2.0,
                    "rationale": "official output",
                    "policy": "opendrivevla_baseline",
                },
            }) + "\n", encoding="utf-8")
            scenario = generate_scenarios(1, 9)[0]
            scenario.scene_id = "cached_scene"
            with patch.dict(os.environ, {"OPENDRIVEVLA_CACHE": str(cache)}):
                baseline = create_base_model(ROOT, "baseline")
                reflected = create_base_model(ROOT, "reflection_sft")
                self.assertEqual(baseline.plan(scenario).target_speed, 2.0)
                self.assertEqual(baseline.metadata()["runtime_mode"], "opendrivevla_cache")
                reflected.plan(scenario)
                self.assertEqual(reflected.metadata()["runtime_mode"], "lite")

    def test_strict_cache_mode_does_not_silently_fallback(self):
        scenario = generate_scenarios(1, 11)[0]
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            os.environ, {"OPENDRIVEVLA_CACHE": str(Path(tmp) / "missing.jsonl")}
        ):
            model = create_base_model(ROOT, "baseline", runtime="cache")
            with self.assertRaises(RuntimeError):
                model.plan(scenario)

    def test_three_file_frontend_exists(self):
        for name in ("index.html", "styles.css", "runtime.css", "app.js"):
            self.assertTrue((ROOT / "demo" / name).is_file())
        self.assertNotIn("const presets=", (ROOT / "demo" / "app.js").read_text(encoding="utf-8"))
        frontend = (ROOT / "demo" / "app.js").read_text(encoding="utf-8")
        page = (ROOT / "demo" / "index.html").read_text(encoding="utf-8")
        self.assertIn("trajectoryHeading", frontend)
        self.assertIn("visual.lead_vehicle.speed * animationTime", frontend)
        self.assertIn("smoothProgress", frontend)
        self.assertNotIn("pollLogs", frontend)
        self.assertNotIn("后端实时日志", page)

    def test_revised_policy_stays_behind_moving_lead_vehicle(self):
        scenario = generate_scenarios(1, 19)[0]
        scenario.ego_speed = 16.0
        scenario.speed_limit = 16.7
        scenario.lead_distance = 8.0
        scenario.lead_speed = 0.5
        scenario.traffic_light = "green"
        scenario.pedestrian_distance = 100.0
        trajectory = Policy("reflection_sft").plan(scenario)
        for index, point in enumerate(trajectory.points, 1):
            lead_x = scenario.lead_distance + scenario.lead_speed * index * .5
            self.assertLessEqual(point[0], lead_x - 3.8 + 1e-6)

    def test_console_log_is_human_readable(self):
        stream = io.StringIO()
        with redirect_stdout(stream):
            RuntimeEventLog().emit(
                "model_call", "规划模型调用完成", policy="reflection_sft",
                actual_runtime="lite", latency_ms=8.2, ignored="verbose",
            )
        output = stream.getvalue()
        self.assertIn("模型：规划模型调用完成", output)
        self.assertIn("policy=reflection_sft", output)
        self.assertNotIn('{"seq"', output)

    def test_demo_indexes_complete_training_file(self):
        status = DATA_STORE.status()
        with (ROOT / "data" / "reflection_dataset.jsonl").open(encoding="utf-8") as handle:
            line_count = sum(1 for line in handle if line.strip())
        self.assertEqual(status["records"], line_count)
        self.assertTrue(status["indexed_all_valid_rows"])
        self.assertGreaterEqual(len(DATA_STORE.presets()), 4)

    def test_nuscenes_records_keep_visual_and_annotation_provenance(self):
        manifest = json.loads((ROOT / "data" / "nuscenes_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["dataset"], "nuScenes")
        self.assertEqual(manifest["records"], 5112)
        with (ROOT / "data" / "reflection_dataset.jsonl").open(encoding="utf-8") as handle:
            row = json.loads(next(handle))
        source = row["source"]
        self.assertEqual(len(source["sample_token"]), 32)
        self.assertEqual(len(source["image_refs"]), 6)
        self.assertTrue(source["annotation_tokens"])
        self.assertEqual(len(row["expert_trajectory"]["points"]), 12)
        self.assertTrue(all((ROOT / "data" / "nuscenes" / path).is_file()
                            for path in source["image_refs"].values()))

    def test_demo_compare_returns_two_trajectories(self):
        obj = {"ego_speed": 13, "speed_limit": 13.9, "lead_distance": 32,
               "lead_speed": 7, "traffic_light": "red", "stopline_distance": 16,
               "pedestrian_distance": 45, "road_curvature": .015,
               "route_command": "straight", "weather": "clear",
               "policy": "reflection_sft"}
        result = compare(obj)
        self.assertIn("baseline", result["results"])
        self.assertIn("reflection_sft", result["results"])
        self.assertEqual(len(result["events"]), 5)
        self.assertEqual(result["results"]["reflection_sft"]["critic"]["critic_type"], "live_rule_reward_data")
        self.assertEqual(result["provenance"]["data"]["records"], DATA_STORE.status()["records"])
        self.assertEqual(len(result["results"]["reflection_sft"]["score_provenance"]["neighbors"]), 7)

    def test_dynamic_demo_http_endpoints(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            with urllib.request.urlopen(base + "/api/data/status") as response:
                status = json.load(response)
            with urllib.request.urlopen(base + "/api/scenarios/presets") as response:
                presets = json.load(response)
            with urllib.request.urlopen(base + "/runtime.css") as response:
                content_type = response.headers.get_content_type()
            image_url = base + presets["items"][0]["camera_image_url"]
            with urllib.request.urlopen(image_url) as response:
                image_type = response.headers.get_content_type()
                image_prefix = response.read(2)
            manifest = json.loads((ROOT / "data" / "nuscenes_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(status["records"], manifest["records"])
            self.assertEqual(len(presets["items"]), 4)
            self.assertEqual(content_type, "text/css")
            self.assertEqual(image_type, "image/jpeg")
            self.assertEqual(image_prefix, b"\xff\xd8")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
