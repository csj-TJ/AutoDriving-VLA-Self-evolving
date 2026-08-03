from __future__ import annotations

import json
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .critics import RewardModelCritic, RuleBasedCritic
from .planner import scenario_features
from .schema import CriticResult, Scenario, Trajectory


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def repository_path(path: Path) -> str:
    """Return a portable path for API/log output without exposing host folders."""
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.name


class RuntimeEventLog:
    """Thread-safe request/model/data trace exposed to the local Demo."""

    def __init__(self, capacity: int = 800):
        self._events: deque[dict[str, Any]] = deque(maxlen=capacity)
        self._lock = threading.Lock()
        self._sequence = 0

    def emit(self, category: str, message: str, **details: Any) -> dict[str, Any]:
        with self._lock:
            self._sequence += 1
            event = {
                "seq": self._sequence,
                "time": datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds"),
                "category": category,
                "message": message,
                "details": details,
            }
            self._events.append(event)
            labels = {
                "request": "请求", "request_complete": "完成", "model_call": "模型",
                "model_load": "模型", "data_load": "数据", "data_warning": "数据",
                "critic_call": "评分", "reflection": "反思", "error": "错误",
            }
            preferred = {
                "request": ("scene_id", "requested_runtime"),
                "request_complete": ("scene_id", "selected_policy", "score_delta"),
                "model_call": ("policy", "actual_runtime", "latency_ms"),
                "model_load": ("path", "latency_ms"),
                "data_load": ("path", "records", "latency_ms"),
                "data_warning": ("line",),
                "critic_call": ("policy", "overall", "latency_ms"),
                "reflection": ("policy", "verdict", "failures"),
            }.get(category, tuple(details)[:3])
            shown = []
            for key in preferred:
                if key not in details:
                    continue
                value = details[key]
                if isinstance(value, list):
                    value = ",".join(map(str, value)) or "无"
                suffix = "ms" if key == "latency_ms" else ""
                shown.append(f"{key}={value}{suffix}")
            clock = event["time"].split("T", 1)[-1][:8]
            tail = " · " + " | ".join(shown) if shown else ""
            print(f"[{clock}] {labels.get(category, category)}：{message}{tail}", flush=True)
            return event

    def read(self, after: int = 0, limit: int = 200) -> dict[str, Any]:
        with self._lock:
            events = [item for item in self._events if item["seq"] > after][-max(1, min(limit, 500)):]
            return {"events": events, "last_seq": self._sequence}


EVENT_LOG = RuntimeEventLog()


class TrainingDataStore:
    """Indexes every valid row and reloads automatically when the JSONL changes."""

    def __init__(self, path: Path):
        self.path = path
        self._mtime_ns = -1
        self._rows: list[dict[str, Any]] = []
        self._by_scene: dict[str, dict[str, Any]] = {}
        self._features = np.empty((0, 9), dtype=float)
        self._scores = np.empty((0, 3), dtype=float)
        self._feature_mean = np.zeros(9, dtype=float)
        self._feature_std = np.ones(9, dtype=float)
        self._lock = threading.RLock()

    @staticmethod
    def _scenario(obj: dict[str, Any]) -> Scenario:
        fields = Scenario.__dataclass_fields__
        return Scenario(**{name: obj[name] for name in fields if name in obj})

    def refresh(self) -> None:
        if not self.path.is_file():
            raise FileNotFoundError(
                f"Training data not found: {self.path}. Confirm data/reflection_dataset.jsonl was pulled."
            )
        mtime_ns = self.path.stat().st_mtime_ns
        if mtime_ns == self._mtime_ns and self._rows:
            return
        started = time.perf_counter()
        rows: list[dict[str, Any]] = []
        features: list[np.ndarray] = []
        scores: list[list[float]] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    scenario = self._scenario(row["scenario"])
                    critic = row["critic"]
                    score = [float(critic[name]) for name in ("safety_score", "rule_score", "comfort_score")]
                except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                    EVENT_LOG.emit("data_warning", "忽略无效训练行", line=line_number)
                    continue
                rows.append(row)
                features.append(scenario_features(scenario))
                scores.append(score)
        if not rows:
            raise ValueError(f"No valid training records in {self.path}")
        matrix = np.vstack(features)
        std = matrix.std(axis=0)
        std[std < 1e-8] = 1.0
        with self._lock:
            self._rows = rows
            self._by_scene = {str(row["scenario"]["scene_id"]): row for row in rows}
            self._features = matrix
            self._scores = np.asarray(scores, dtype=float)
            self._feature_mean = matrix.mean(axis=0)
            self._feature_std = std
            self._mtime_ns = mtime_ns
        EVENT_LOG.emit(
            "data_load", "完整训练数据索引已加载", path=repository_path(self.path), records=len(rows),
            bytes=self.path.stat().st_size, latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )

    def status(self) -> dict[str, Any]:
        self.refresh()
        with self._lock:
            splits: dict[str, int] = {}
            for row in self._rows:
                key = str(row.get("split", "unknown"))
                splits[key] = splits.get(key, 0) + 1
            return {
                "path": repository_path(self.path), "records": len(self._rows), "bytes": self.path.stat().st_size,
                "mtime_ns": self._mtime_ns, "splits": splits, "indexed_all_valid_rows": True,
            }

    def get(self, scene_id: str) -> dict[str, Any]:
        self.refresh()
        try:
            return self._by_scene[scene_id]
        except KeyError as exc:
            raise KeyError(f"Unknown training scene: {scene_id}") from exc

    def page(self, offset: int = 0, limit: int = 50, query: str = "") -> dict[str, Any]:
        self.refresh()
        query = query.strip().lower()
        rows = self._rows
        if query:
            rows = [row for row in rows if query in str(row.get("sample_id", "")).lower()
                    or query in str(row["scenario"].get("scene_id", "")).lower()]
        offset = max(0, offset)
        limit = max(1, min(limit, 200))
        items = [{
            "sample_id": row.get("sample_id"), "scene_id": row["scenario"]["scene_id"],
            "split": row.get("split"), "accepted": row.get("accepted"),
            "failures": row.get("critic", {}).get("failures", []), "scenario": row["scenario"],
            "camera_image_url": f"/api/nuscenes/image?scene_id={row['scenario']['scene_id']}&camera=CAM_FRONT"
            if row.get("source", {}).get("dataset") == "nuScenes" else None,
        } for row in rows[offset:offset + limit]]
        return {"items": items, "offset": offset, "limit": limit, "total": len(rows)}

    def presets(self) -> list[dict[str, Any]]:
        self.refresh()
        definitions = (
            ("intersection", "真实路口", lambda r: "intersection" in r.get("source", {}).get("scene_description", "").lower()),
            ("ped", "行人横穿", lambda r: r["scenario"].get("pedestrian_distance", 100) < 18),
            ("lead", "近距前车", lambda r: r["scenario"].get("lead_distance", 100) < 12),
            ("curve", "转弯场景", lambda r: r["scenario"].get("route_command") in {"left", "right"}),
        )
        result = []
        for key, label, predicate in definitions:
            row = next((item for item in self._rows if predicate(item)), None)
            if row:
                result.append({"key": key, "label": label, "sample_id": row.get("sample_id"),
                               "scene_id": row["scenario"]["scene_id"], "scenario": row["scenario"],
                               "camera_image_url": f"/api/nuscenes/image?scene_id={row['scenario']['scene_id']}&camera=CAM_FRONT"
                               if row.get("source", {}).get("dataset") == "nuScenes" else None})
        return result

    def nearest(self, scenario: Scenario, k: int = 7) -> dict[str, Any]:
        self.refresh()
        feature = scenario_features(scenario)
        distances = np.linalg.norm((self._features - feature) / self._feature_std, axis=1)
        count = min(max(1, k), len(distances))
        indices = np.argpartition(distances, count - 1)[:count]
        indices = indices[np.argsort(distances[indices])]
        neighbor_scores = self._scores[indices]
        items = [{
            "sample_id": self._rows[int(i)].get("sample_id"),
            "scene_id": self._rows[int(i)]["scenario"]["scene_id"],
            "distance": round(float(distances[int(i)]), 4),
        } for i in indices]
        return {"k": count, "neighbors": items, "mean_scores": neighbor_scores.mean(axis=0).tolist()}


class LiveDataCritic:
    """Fresh trajectory scoring with rule, trained reward model, and full-data calibration."""

    def __init__(self, root: Path, data_store: TrainingDataStore):
        config = json.loads((root / "configs" / "default.json").read_text(encoding="utf-8"))
        self.rule = RuleBasedCritic(config["critic_weights"])
        self.reward_path = root / "models" / "reward_critic.json"
        self.reward: RewardModelCritic | None = None
        self._reward_mtime_ns = -1
        self.data_store = data_store

    def _refresh_reward(self) -> RewardModelCritic:
        mtime_ns = self.reward_path.stat().st_mtime_ns
        if self.reward is None or mtime_ns != self._reward_mtime_ns:
            started = time.perf_counter()
            reward_obj = json.loads(self.reward_path.read_text(encoding="utf-8"))
            self.reward = RewardModelCritic(reward_obj["coefficients"])
            self._reward_mtime_ns = mtime_ns
            EVENT_LOG.emit(
                "model_load", "Reward Critic 参数文件已连接", path=repository_path(self.reward_path),
                coefficient_shape=list(self.reward.coef.shape),
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
            )
        return self.reward

    def evaluate(self, scenario: Scenario, trajectory: Trajectory) -> tuple[CriticResult, dict[str, Any]]:
        started = time.perf_counter()
        rule = self.rule.evaluate(scenario, trajectory)
        reward = self._refresh_reward().evaluate(scenario, trajectory)
        context = self.data_store.nearest(scenario)
        prior = context["mean_scores"]
        components = (
            (.65, [rule.safety_score, rule.rule_score, rule.comfort_score]),
            (.25, [reward.safety_score, reward.rule_score, reward.comfort_score]),
            (.10, prior),
        )
        scores = [round(sum(weight * values[i] for weight, values in components), 3) for i in range(3)]
        overall = round(.45 * scores[0] + .35 * scores[1] + .20 * scores[2], 3)
        evidence = list(rule.evidence)
        evidence.append(
            f"实时评分：规则65% + Reward Critic25% + 全训练集{context['k']}近邻校准10%"
        )
        result = CriticResult(
            "live_rule_reward_data", scores[0], scores[1], scores[2], overall,
            rule.failures, evidence,
        )
        provenance = {
            "rule_scores": [rule.safety_score, rule.rule_score, rule.comfort_score],
            "reward_scores": [reward.safety_score, reward.rule_score, reward.comfort_score],
            "training_prior_scores": [round(float(v), 3) for v in prior],
            "reward_model_file": repository_path(self.reward_path),
            "training_data_file": repository_path(self.data_store.path),
            "neighbors": context["neighbors"],
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }
        return result, provenance
