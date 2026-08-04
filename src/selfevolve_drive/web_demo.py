from __future__ import annotations

import json
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .base_models import audit_opendrivevla, create_base_model
from .demo_runtime import EVENT_LOG, LiveDataCritic, TrainingDataStore
from .driving_skills import build_driving_skill
from .reflection import reflect
from .schema import Scenario

ROOT = Path(__file__).resolve().parents[2]


DATA_STORE = TrainingDataStore(ROOT / "data" / "reflection_dataset.jsonl")
CRITIC = LiveDataCritic(ROOT, DATA_STORE)


def evolution_history() -> list[dict]:
    path = ROOT / "outputs" / "evolution_history.json"
    if not path.is_file():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return []


def _scenario(obj: dict) -> Scenario:
    return Scenario(
        scene_id=str(obj.get("scene_id", "demo")),
        ego_speed=float(obj["ego_speed"]), speed_limit=float(obj["speed_limit"]),
        lead_distance=float(obj["lead_distance"]), lead_speed=float(obj["lead_speed"]),
        traffic_light=obj["traffic_light"], stopline_distance=float(obj["stopline_distance"]),
        pedestrian_distance=float(obj["pedestrian_distance"]), road_curvature=float(obj["road_curvature"]),
        route_command=obj["route_command"], weather=obj["weather"],
        unseen=bool(obj.get("unseen", False)),
    )


def _source_record(obj: dict) -> dict | None:
    if obj.get("scenario_source") == "user_control_modified":
        return None
    record_id = obj.get("source_sample_id") or obj.get("scene_id")
    if not record_id:
        return None
    try:
        return DATA_STORE.get(str(record_id))
    except KeyError:
        return None


def _visualization(s: Scenario, source_record: dict | None = None) -> dict:
    def center_y(x: float) -> float:
        return s.road_curvature * (max(0.0, x) ** 1.45) * .12

    horizon = max(70.0, s.stopline_distance + 15.0, min(s.lead_distance + 15.0, 100.0))
    centerline = [[round(x, 3), round(center_y(x), 3)] for x in
                  [i * horizon / 35 for i in range(36)]]
    pedestrian_track = (
        source_record.get("source", {}).get("participants", {}).get("pedestrian")
        if source_record else None
    )
    if pedestrian_track and pedestrian_track.get("points"):
        pedestrian = {
            "visible": True, "track": pedestrian_track["points"],
            "data_source": "nuscenes_sample_annotation",
            "annotation_token": pedestrian_track.get("annotation_token"),
            "instance_token": pedestrian_track.get("instance_token"),
            "duration_s": pedestrian_track.get("duration_s", 0.0),
            "displacement_m": pedestrian_track.get("displacement_m", 0.0),
            "mean_speed_mps": pedestrian_track.get("mean_speed_mps", 0.0),
        }
    else:
        visible = s.pedestrian_distance <= horizon
        pedestrian = {
            "visible": visible,
            "track": ([{"t": 0.0, "x": s.pedestrian_distance,
                        "y": center_y(s.pedestrian_distance)}] if visible else []),
            "data_source": "scenario_static" if visible else "none",
            "duration_s": 0.0, "displacement_m": 0.0, "mean_speed_mps": 0.0,
        }
    payload = {
        "world_horizon_m": horizon,
        "road_width_m": 10.5,
        "lane_count": 3,
        "centerline": centerline,
        "stop_line": {"x": s.stopline_distance, "center_y": center_y(s.stopline_distance)},
        "lead_vehicle": {"x": s.lead_distance, "y": center_y(s.lead_distance),
                         "speed": s.lead_speed, "visible": s.lead_distance <= horizon},
        "pedestrian": pedestrian,
        "traffic_light": {"state": s.traffic_light, "x": s.stopline_distance,
                          "y": center_y(s.stopline_distance) + 6.5},
        "route_command": s.route_command,
        "weather": s.weather,
    }
    return payload


def _run_policy(s: Scenario, name: str, runtime: str, request_id: str) -> tuple[dict, dict]:
    model = create_base_model(ROOT, name, runtime)
    started = time.perf_counter()
    trajectory = model.plan(s)
    model_meta = model.metadata()
    model_latency = round((time.perf_counter() - started) * 1000, 2)
    EVENT_LOG.emit(
        "model_call", "规划模型调用完成", request_id=request_id, scene_id=s.scene_id,
        policy=name, requested_runtime=runtime, actual_runtime=model_meta.get("runtime_mode"),
        weights_loaded=model_meta.get("weights_loaded"), latency_ms=model_latency,
    )
    critic, score_source = CRITIC.evaluate(s, trajectory)
    EVENT_LOG.emit(
        "critic_call", "Critic 实时评分完成", request_id=request_id, scene_id=s.scene_id,
        policy=name, critic_type=critic.critic_type, overall=critic.overall_score,
        training_neighbors=[item["sample_id"] for item in score_source["neighbors"]],
        latency_ms=score_source["latency_ms"],
    )
    reflection = reflect(s, trajectory, critic)
    EVENT_LOG.emit(
        "reflection", "结构化反思完成", request_id=request_id, policy=name,
        verdict=reflection.verdict, failures=critic.failures,
    )
    return {
        "trajectory": trajectory.to_dict(), "critic": critic.to_dict(),
        "reflection": reflection.to_dict(), "model": model_meta,
        "score_provenance": score_source,
    }, {"model_latency_ms": model_latency, "actual_runtime": model_meta.get("runtime_mode")}


def infer(obj: dict) -> dict:
    s = _scenario(obj)
    source_record = _source_record(obj)
    request_id = str(obj.get("request_id") or uuid.uuid4().hex[:12])
    policy = obj.get("policy", "reflection_sft")
    result, timing = _run_policy(s, policy, obj.get("runtime", "auto"), request_id)
    payload = {
        "request_id": request_id, "scenario": s.to_dict(), "visualization": _visualization(s, source_record),
        **result, "provenance": {"scenario_source": obj.get("scenario_source", "user_control"),
                                "source_sample_id": obj.get("source_sample_id"),
                                "data": DATA_STORE.status(), "timing": timing},
    }
    return payload


def compare(obj: dict) -> dict:
    s = _scenario(obj)
    source_record = _source_record(obj)
    request_id = str(obj.get("request_id") or uuid.uuid4().hex[:12])
    runtime = obj.get("runtime", "auto")
    EVENT_LOG.emit(
        "request", "开始双策略实时评估", request_id=request_id, scene_id=s.scene_id,
        scenario_source=obj.get("scenario_source", "user_control"),
        source_sample_id=obj.get("source_sample_id"), requested_runtime=runtime,
    )
    results = {}
    timings = {}
    for name in ("baseline", obj.get("policy", "reflection_sft")):
        results[name], timings[name] = _run_policy(s, name, runtime, request_id)
    selected = obj.get("policy", "reflection_sft")
    base_score = results["baseline"]["critic"]["overall_score"]
    improved_score = results[selected]["critic"]["overall_score"]
    failures = results["baseline"]["critic"]["failures"]
    generated_skill = build_driving_skill(
        s, results["baseline"], results[selected],
        ROOT / "outputs" / "reflection_memory.jsonl", source_record,
    )
    events = [
        {"phase": "感知", "detail": f"检测到前车 {s.lead_distance:.0f}m、行人 {s.pedestrian_distance:.0f}m、{s.traffic_light} 信号灯"},
        {"phase": "初始规划", "detail": f"Baseline 目标速度 {results['baseline']['trajectory']['target_speed']:.1f} m/s"},
        {"phase": "Critic", "detail": "；".join(results["baseline"]["critic"]["evidence"]) or "未发现硬约束失败"},
        {"phase": "反思", "detail": "；".join(results["baseline"]["reflection"]["corrective_strategy"]) or "保持当前安全策略"},
        {"phase": "Skill生成", "detail": f"{generated_skill['skill_id']} · {generated_skill['name']} · 历史支持 {generated_skill['memory']['matched_records']} 条"},
        {"phase": "重规划", "detail": f"{selected} 目标速度 {results[selected]['trajectory']['target_speed']:.1f} m/s，综合分变化 {improved_score-base_score:+.1f}"},
    ]
    payload = {
        "request_id": request_id, "scenario": s.to_dict(), "visualization": _visualization(s, source_record),
        "results": results, "selected_policy": selected,
        "events": events, "failure_count": len(failures),
        "delta": {"overall": round(improved_score - base_score, 3), "target_speed": round(results[selected]["trajectory"]["target_speed"] - results["baseline"]["trajectory"]["target_speed"], 3)},
        "model": results[selected]["model"],
        "evolution_history": evolution_history(),
        "generated_skill": generated_skill,
        "provenance": {
            "scenario_source": obj.get("scenario_source", "user_control"),
            "source_sample_id": obj.get("source_sample_id"),
            "data": DATA_STORE.status(), "timing": timings,
            "score_formula": "65% fresh rule + 25% trained reward critic + 10% full-dataset kNN prior",
        },
    }
    EVENT_LOG.emit(
        "request_complete", "双策略评估已返回", request_id=request_id, scene_id=s.scene_id,
        selected_policy=selected, score_delta=payload["delta"]["overall"],
    )
    return payload


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)
        if path == "/api/logs":
            self._json(EVENT_LOG.read(
                after=int(query.get("after", [0])[0]), limit=int(query.get("limit", [200])[0])))
            return
        if path == "/api/data/status":
            self._json(DATA_STORE.status())
            return
        if path == "/api/scenarios/presets":
            presets = DATA_STORE.presets()
            self._json({"items": presets, "data": DATA_STORE.status()})
            return
        if path == "/api/scenarios":
            self._json(DATA_STORE.page(
                offset=int(query.get("offset", [0])[0]), limit=int(query.get("limit", [50])[0]),
                query=query.get("query", [""])[0]))
            return
        if path == "/api/scenario":
            scene_id = query.get("id", [""])[0]
            row = DATA_STORE.get(scene_id)
            scenario = _scenario(row["scenario"])
            EVENT_LOG.emit("data_select", "从训练数据加载场景", scene_id=scene_id,
                           sample_id=row.get("sample_id"), split=row.get("split"))
            self._json({"sample_id": row.get("sample_id"), "scene_id": scene_id,
                        "scenario": scenario.to_dict(), "visualization": _visualization(scenario, row),
                        "stored_critic": row.get("critic"), "split": row.get("split"),
                        "source": row.get("source"),
                        "camera_image_url": f"/api/nuscenes/image?scene_id={scene_id}&camera=CAM_FRONT"
                        if row.get("source", {}).get("dataset") == "nuScenes" else None})
            return
        if path == "/api/nuscenes/image":
            scene_id = query.get("scene_id", [""])[0]
            camera = query.get("camera", ["CAM_FRONT"])[0]
            row = DATA_STORE.get(scene_id)
            relative = row.get("source", {}).get("image_refs", {}).get(camera)
            if not relative:
                self.send_error(404); return
            media_root = (ROOT / "data" / "nuscenes").resolve()
            image_path = (media_root / relative).resolve()
            if media_root not in image_path.parents or not image_path.is_file():
                self.send_error(404); return
            data = image_path.read_bytes()
            self.send_response(200); self.send_header("Content-Type", "image/jpeg")
            self.send_header("Cache-Control", "public, max-age=3600")
            self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
            return
        if path == "/api/meta":
            self._json({"model": create_base_model(ROOT).metadata(), "audit": audit_opendrivevla(ROOT),
                        "data": DATA_STORE.status(), "evolution_history": evolution_history(),
                        "policies": ["sft", "reflection_sft", "reflection_dpo"]})
            return
        if path == "/api/audit":
            self._json(audit_opendrivevla(ROOT))
            return
        if path == "/api/evolution":
            self._json({"history": evolution_history()})
            return
        if path == "/api/model/status":
            audit = audit_opendrivevla(ROOT)
            self._json({
                "checkpoint": {
                    "installed": audit["checkpoint_installed"],
                    "path": audit["model_dir"],
                    "size_bytes": audit["checkpoint_size_bytes"],
                    "architecture": audit["architecture"],
                    "safetensors_header_ok": audit["safetensors_header_ok"],
                },
                "official_code": {"installed": audit["code_complete"], "path": audit["code_dir"]},
                "runtime": {
                    "live_gpu_ready": audit["runtime_ready"],
                    "cache_available": audit["cache_available"],
                    "cache_path": audit["cache_path"],
                    "available_modes": ["auto", "cache", "lite"],
                },
            })
            return
        static_files = {
            "/": ("index.html", "text/html; charset=utf-8"),
            "/index.html": ("index.html", "text/html; charset=utf-8"),
            "/styles.css": ("styles.css", "text/css; charset=utf-8"),
            "/runtime.css": ("runtime.css", "text/css; charset=utf-8"),
            "/app.js": ("app.js", "text/javascript; charset=utf-8"),
        }
        if path not in static_files:
            self.send_error(404); return
        filename, content_type = static_files[path]
        data = (ROOT / "demo" / filename).read_bytes()
        self.send_response(200); self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)

    def do_POST(self):
        path = urlparse(self.path).path
        if path not in ("/api/evaluate", "/api/model/plan", "/api/compare"): self.send_error(404); return
        try:
            size = int(self.headers.get("Content-Length", "0")); obj = json.loads(self.rfile.read(size))
            self._json(compare(obj) if path == "/api/compare" else infer(obj))
        except Exception as exc:
            EVENT_LOG.emit("error", "API 调用失败", path=path, error=str(exc))
            self._json({"error": str(exc)}, status=400)

    def _json(self, obj: dict, status: int = 200):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status); self.send_header("Content-Type", "application/json; charset=utf-8"); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)

    def log_message(self, fmt, *args):
        pass


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    print(f"Demo: http://{host}:{port}")
    ThreadingHTTPServer((host, port), Handler).serve_forever()
