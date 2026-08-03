from __future__ import annotations

from collections import Counter
from typing import Any
import numpy as np

from .critics import RuleBasedCritic
from .planner import Policy
from .schema import Scenario
from .simulator import expert_target_speed


def evaluate_policy(
    policy: Policy,
    scenes: list[Scenario],
    weights: dict[str, float],
    target_speeds: list[float] | None = None,
) -> dict[str, Any]:
    critic = RuleBasedCritic(weights)
    scores, failures, errors = [], Counter(), []
    for index, s in enumerate(scenes):
        t = policy.plan(s)
        c = critic.evaluate(s, t)
        scores.append([c.safety_score, c.rule_score, c.comfort_score, c.overall_score])
        failures.update(c.failures)
        expected = target_speeds[index] if target_speeds is not None else expert_target_speed(s)
        errors.append(abs(t.target_speed - expected))
    a = np.asarray(scores)
    n = max(1, len(scenes))
    return {
        "policy": policy.name, "samples": len(scenes),
        "collision_risk_rate": round(failures["unsafe_following"] / n, 4),
        "traffic_violation_rate": round((failures["speeding"] + failures["red_light_risk"]) / n, 4),
        "pedestrian_yield_failure_rate": round(failures["pedestrian_yield_failure"] / n, 4),
        "uncomfortable_rate": round(failures["uncomfortable_motion"] / n, 4),
        "safety_score": round(float(a[:, 0].mean()), 3), "rule_score": round(float(a[:, 1].mean()), 3),
        "comfort_score": round(float(a[:, 2].mean()), 3), "overall_score": round(float(a[:, 3].mean()), 3),
        "target_speed_mae": round(float(np.mean(errors)), 3),
    }


def critic_agreement(records: list[dict]) -> dict[str, Any]:
    by_type: dict[str, list[float]] = {}
    for r in records:
        by_type.setdefault(r["critic"]["critic_type"], []).append(r["critic"]["overall_score"])
    return {k: {"count": len(v), "mean_overall": round(float(np.mean(v)), 3), "std": round(float(np.std(v)), 3)} for k, v in by_type.items()}
