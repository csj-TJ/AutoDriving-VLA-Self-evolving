from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from .base_models import audit_opendrivevla, create_base_model
from .critics import RuleBasedCritic
from .reflection import reflect
from .schema import Scenario

ROOT = Path(__file__).resolve().parents[2]


CRITIC = RuleBasedCritic({"safety": .45, "rule": .35, "comfort": .20})


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
    )


def infer(obj: dict) -> dict:
    s = _scenario(obj)
    model = create_base_model(ROOT, obj.get("policy", "reflection_sft"), obj.get("runtime", "auto"))
    t = model.plan(s)
    c = CRITIC.evaluate(s, t)
    r = reflect(s, t, c)
    return {"scenario": s.to_dict(), "trajectory": t.to_dict(), "critic": c.to_dict(), "reflection": r.to_dict(), "model": model.metadata()}


def compare(obj: dict) -> dict:
    s = _scenario(obj)
    results = {}
    models = {}
    for name in ("baseline", obj.get("policy", "reflection_sft")):
        model = create_base_model(ROOT, name, obj.get("runtime", "auto"))
        t = model.plan(s)
        c = CRITIC.evaluate(s, t)
        results[name] = {"trajectory": t.to_dict(), "critic": c.to_dict(), "reflection": reflect(s, t, c).to_dict(), "model": model.metadata()}
        models[name] = model
    selected = obj.get("policy", "reflection_sft")
    base_score = results["baseline"]["critic"]["overall_score"]
    improved_score = results[selected]["critic"]["overall_score"]
    failures = results["baseline"]["critic"]["failures"]
    events = [
        {"phase": "感知", "detail": f"检测到前车 {s.lead_distance:.0f}m、行人 {s.pedestrian_distance:.0f}m、{s.traffic_light} 信号灯"},
        {"phase": "初始规划", "detail": f"Baseline 目标速度 {results['baseline']['trajectory']['target_speed']:.1f} m/s"},
        {"phase": "Critic", "detail": "；".join(results["baseline"]["critic"]["evidence"]) or "未发现硬约束失败"},
        {"phase": "反思", "detail": "；".join(results["baseline"]["reflection"]["corrective_strategy"]) or "保持当前安全策略"},
        {"phase": "重规划", "detail": f"{selected} 目标速度 {results[selected]['trajectory']['target_speed']:.1f} m/s，综合分变化 {improved_score-base_score:+.1f}"},
    ]
    return {
        "scenario": s.to_dict(), "results": results, "selected_policy": selected,
        "events": events, "failure_count": len(failures),
        "delta": {"overall": round(improved_score - base_score, 3), "target_speed": round(results[selected]["trajectory"]["target_speed"] - results["baseline"]["trajectory"]["target_speed"], 3)},
        "model": models[selected].metadata(),
        "evolution_history": evolution_history(),
    }


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/meta":
            self._json({"model": create_base_model(ROOT).metadata(), "audit": audit_opendrivevla(ROOT), "evolution_history": evolution_history(), "policies": ["sft", "reflection_sft", "reflection_dpo"]})
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
            self._json({"error": str(exc)}, status=400)

    def _json(self, obj: dict, status: int = 200):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status); self.send_header("Content-Type", "application/json; charset=utf-8"); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)

    def log_message(self, fmt, *args):
        pass


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    print(f"Demo: http://{host}:{port}")
    ThreadingHTTPServer((host, port), Handler).serve_forever()
