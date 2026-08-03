import sys
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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
from selfevolve_drive.web_demo import compare


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
        for name in ("index.html", "styles.css", "app.js"):
            self.assertTrue((ROOT / "demo" / name).is_file())

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


if __name__ == "__main__":
    unittest.main()
