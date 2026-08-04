from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from typing import Any

import numpy as np

from .planner import scenario_features
from .schema import CriticResult, Scenario, Trajectory
from .simulator import assess_pedestrian_conflict


def _clip(x: float) -> float:
    return float(round(max(0.0, min(100.0, float(x))), 3))


def critic_features(s: Scenario, t: Trajectory) -> np.ndarray:
    pts = np.asarray(t.points, dtype=float)
    speeds = pts[:, 2]
    acc = np.diff(speeds, prepend=s.ego_speed) / 0.5
    jerk = np.diff(acc, prepend=acc[0]) / 0.5
    lateral = pts[:, 1]
    lat_acc = np.diff(np.diff(lateral, prepend=0.0), prepend=0.0) / (0.5 ** 2)
    return np.concatenate([scenario_features(s), [t.target_speed, np.max(speeds), np.mean(np.abs(jerk)), np.max(np.abs(lat_acc))]])


@dataclass
class RuleBasedCritic:
    weights: dict[str, float]

    def evaluate(self, s: Scenario, t: Trajectory) -> CriticResult:
        pts = np.asarray(t.points, dtype=float)
        speeds = pts[:, 2]
        failures: list[str] = []
        evidence: list[str] = []
        safety = 100.0
        rule = 100.0

        closing = max(0.0, float(np.max(speeds)) - s.lead_speed)
        ttc = s.lead_distance / max(closing, 0.1)
        if ttc < 3.0:
            safety -= (3.0 - ttc) * 28.0
            failures.append("unsafe_following")
            evidence.append(f"minimum TTC proxy={ttc:.2f}s < 3.0s")
        pedestrian = assess_pedestrian_conflict(s, trajectory_points=t.points)
        if pedestrian.source == "legacy_distance":
            unsafe_legacy_speed = max(1.0, (s.pedestrian_distance - 3.0) / 2.5 + 1.2)
            if pedestrian.relevant and t.target_speed > unsafe_legacy_speed:
                safety -= 55.0
                failures.append("pedestrian_yield_failure")
                evidence.append(
                    f"pedestrian at {s.pedestrian_distance:.1f}m while target speed={t.target_speed:.1f}m/s"
                )
        elif pedestrian.relevant:
            safety -= 55.0
            failures.append("pedestrian_yield_failure")
            evidence.append(
                f"预测 {pedestrian.conflict_time_s:.1f}s 时车辆与行人最小间距仅 "
                f"{pedestrian.minimum_separation_m:.1f}m"
            )
        elif pedestrian.minimum_separation_m is not None:
            evidence.append(
                f"行人与车辆轨迹无时空冲突，预测最小间距 "
                f"{pedestrian.minimum_separation_m:.1f}m"
            )
        if float(np.max(speeds)) > s.speed_limit + 0.8:
            rule -= min(60.0, (float(np.max(speeds)) - s.speed_limit) * 12.0)
            failures.append("speeding")
            evidence.append(f"peak speed={np.max(speeds):.1f}m/s exceeds limit={s.speed_limit:.1f}m/s")
        stopping_speed = max(0.0, (s.stopline_distance - 2.5) / 3.0)
        if s.traffic_light == "red" and t.target_speed > stopping_speed + 1.0:
            rule -= 65.0
            failures.append("red_light_risk")
            evidence.append(f"red light at {s.stopline_distance:.1f}m; target speed too high for stop profile")

        acc = np.diff(speeds, prepend=s.ego_speed) / 0.5
        jerk = np.diff(acc, prepend=acc[0]) / 0.5
        lateral = pts[:, 1]
        lat_acc = np.diff(np.diff(lateral, prepend=0.0), prepend=0.0) / (0.5 ** 2)
        max_jerk = float(np.max(np.abs(jerk)))
        max_lat = float(np.max(np.abs(lat_acc)))
        comfort = 100.0 - max(0.0, max_jerk - 2.0) * 12.0 - max(0.0, max_lat - 2.5) * 9.0
        if comfort < 75.0:
            failures.append("uncomfortable_motion")
            evidence.append(f"max jerk={max_jerk:.2f}m/s^3, max lateral acceleration proxy={max_lat:.2f}m/s^2")
        safety, rule, comfort = _clip(safety), _clip(rule), _clip(comfort)
        overall = _clip(self.weights["safety"] * safety + self.weights["rule"] * rule + self.weights["comfort"] * comfort)
        return CriticResult("rule", safety, rule, comfort, overall, sorted(set(failures)), evidence)


class LLMCritic:
    """OpenAI-compatible Critic with a deterministic offline fallback for demos."""

    def __init__(self, rule_critic: RuleBasedCritic, config: dict[str, Any]):
        self.rule_critic = rule_critic
        self.config = config

    def evaluate(self, s: Scenario, t: Trajectory) -> CriticResult:
        base = self.rule_critic.evaluate(s, t)
        if self.config.get("mode") != "online":
            # Reproducible proxy: emulate bounded disagreement while retaining explanations.
            delta = ((sum(map(ord, s.scene_id)) % 9) - 4) * 0.7
            scores = [_clip(base.safety_score + delta), _clip(base.rule_score - delta / 2), _clip(base.comfort_score + delta / 3)]
            overall = _clip(sum(w * v for w, v in zip((0.45, 0.35, 0.20), scores)))
            return CriticResult("llm_offline_proxy", *scores, overall, base.failures, base.evidence + ["offline deterministic LLM-Critic proxy"])
        payload = {"model": self.config["model"], "messages": [{"role": "user", "content": self.prompt(s, t)}], "temperature": 0, "response_format": {"type": "json_object"}}
        req = urllib.request.Request(self.config["base_url"].rstrip("/") + "/chat/completions", data=json.dumps(payload).encode(), headers={"Authorization": "Bearer " + os.environ[self.config["api_key_env"]], "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as response:
            raw = json.loads(response.read())["choices"][0]["message"]["content"]
        obj = json.loads(raw)
        return CriticResult("llm", obj["safety_score"], obj["rule_score"], obj["comfort_score"], obj["overall_score"], obj.get("failures", []), obj.get("evidence", []))

    @staticmethod
    def prompt(s: Scenario, t: Trajectory) -> str:
        return "Evaluate this driving plan on safety, traffic-rule compliance, and comfort (0-100). Return strict JSON.\n" + json.dumps({"scenario": s.to_dict(), "trajectory": t.to_dict()}, ensure_ascii=False)


class RewardModelCritic:
    def __init__(self, coefficients: list[list[float]]):
        self.coef = np.asarray(coefficients, dtype=float)

    @classmethod
    def fit(cls, examples: list[tuple[Scenario, Trajectory, CriticResult]], ridge: float = 0.1) -> "RewardModelCritic":
        x = np.vstack([critic_features(s, t) for s, t, _ in examples])
        x = np.column_stack([np.ones(len(x)), x])
        y = np.asarray([[r.safety_score, r.rule_score, r.comfort_score] for _, _, r in examples])
        coef = np.linalg.solve(x.T @ x + ridge * np.eye(x.shape[1]), x.T @ y)
        return cls(coef.tolist())

    def evaluate(self, s: Scenario, t: Trajectory) -> CriticResult:
        x = np.concatenate([[1.0], critic_features(s, t)])
        scores = [_clip(v) for v in x @ self.coef]
        overall = _clip(0.45 * scores[0] + 0.35 * scores[1] + 0.20 * scores[2])
        return CriticResult("reward_model", *scores, overall, [], ["scores predicted by ridge reward surrogate"])

    def to_dict(self) -> dict[str, Any]:
        return {"coefficients": self.coef.tolist()}
