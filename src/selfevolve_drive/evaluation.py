from __future__ import annotations

from collections import Counter
from typing import Any
import numpy as np

from .critics import RuleBasedCritic
from .planner import Policy
from .schema import Scenario
from .schema import Trajectory
from .simulator import expert_target_speed


def evaluate_policy(
    policy: Policy,
    scenes: list[Scenario],
    weights: dict[str, float],
    target_speeds: list[float] | None = None,
    expert_trajectories: list[Trajectory] | None = None,
    sample_tokens: list[str] | None = None,
) -> dict[str, Any]:
    critic = RuleBasedCritic(weights)
    scores, failures, errors, l2_rows, jerks = [], Counter(), [], [], []
    per_sample: dict[str, list[float]] = {}
    scenario_groups: dict[str, list[list[float]]] = {
        "pedestrian": [], "close_lead": [], "turn": [], "adverse_weather": [], "all": [],
    }
    for index, s in enumerate(scenes):
        t = policy.plan(s)
        c = critic.evaluate(s, t)
        scores.append([c.safety_score, c.rule_score, c.comfort_score, c.overall_score])
        failures.update(c.failures)
        expected = target_speeds[index] if target_speeds is not None else expert_target_speed(s)
        errors.append(abs(t.target_speed - expected))
        pts = np.asarray(t.points, dtype=float)
        acceleration = np.diff(pts[:, 2], prepend=s.ego_speed) / 0.5
        jerks.append(float(np.mean(np.abs(np.diff(acceleration, prepend=acceleration[0]) / 0.5))))
        if expert_trajectories is not None:
            expert = np.asarray(expert_trajectories[index].points, dtype=float)
            count = min(len(pts), len(expert))
            l2 = np.linalg.norm(pts[:count, :2] - expert[:count, :2], axis=1)
            l2_rows.append(l2)
        token = sample_tokens[index] if sample_tokens is not None else str(index)
        per_sample.setdefault(token, []).append(c.overall_score)
        row = [c.overall_score, float("unsafe_following" in c.failures),
               float("pedestrian_yield_failure" in c.failures)]
        scenario_groups["all"].append(row)
        if s.pedestrian_distance < 60.0: scenario_groups["pedestrian"].append(row)
        if s.lead_distance < 30.0: scenario_groups["close_lead"].append(row)
        if s.route_command in {"left", "right"}: scenario_groups["turn"].append(row)
        if s.weather in {"rain", "fog", "night"}: scenario_groups["adverse_weather"].append(row)
    a = np.asarray(scores)
    n = max(1, len(scenes))
    cluster_means = np.asarray([np.mean(values) for values in per_sample.values()], dtype=float)
    ci_half = 1.96 * float(np.std(cluster_means, ddof=1)) / max(1.0, np.sqrt(len(cluster_means))) if len(cluster_means) > 1 else 0.0
    result = {
        "policy": policy.name, "samples": len(scenes),
        "unique_keyframes": len(per_sample),
        "collision_risk_rate": round(failures["unsafe_following"] / n, 4),
        "traffic_violation_rate": round((failures["speeding"] + failures["red_light_risk"]) / n, 4),
        "pedestrian_yield_failure_rate": round(failures["pedestrian_yield_failure"] / n, 4),
        "uncomfortable_rate": round(failures["uncomfortable_motion"] / n, 4),
        "safety_score": round(float(a[:, 0].mean()), 3), "rule_score": round(float(a[:, 1].mean()), 3),
        "comfort_score": round(float(a[:, 2].mean()), 3), "overall_score": round(float(a[:, 3].mean()), 3),
        "target_speed_mae": round(float(np.mean(errors)), 3),
        "mean_abs_jerk": round(float(np.mean(jerks)), 3),
        "overall_95ci": [round(float(a[:, 3].mean()) - ci_half, 3), round(float(a[:, 3].mean()) + ci_half, 3)],
        "scenario_breakdown": {
            name: {
                "samples": len(rows),
                "overall_score": round(float(np.mean([row[0] for row in rows])), 3) if rows else None,
                "collision_risk_rate": round(float(np.mean([row[1] for row in rows])), 4) if rows else None,
                "pedestrian_failure_rate": round(float(np.mean([row[2] for row in rows])), 4) if rows else None,
            } for name, rows in scenario_groups.items()
        },
    }
    if l2_rows:
        l2_array = np.vstack(l2_rows)
        indices = [min(l2_array.shape[1] - 1, i) for i in (1, 3, 5)]
        result["planning_l2_m"] = {
            "1s": round(float(l2_array[:, indices[0]].mean()), 3),
            "2s": round(float(l2_array[:, indices[1]].mean()), 3),
            "3s": round(float(l2_array[:, indices[2]].mean()), 3),
            "avg_1_2_3s": round(float(l2_array[:, indices].mean()), 3),
            "ade_3s": round(float(l2_array[:, :indices[2] + 1].mean()), 3),
        }
    return result


def critic_agreement(records: list[dict]) -> dict[str, Any]:
    by_type: dict[str, list[tuple[float, float, bool]]] = {}
    rule = RuleBasedCritic({"safety": .45, "rule": .35, "comfort": .20})
    for r in records:
        scenario = Scenario(**r["scenario"])
        trajectory = Trajectory(**r["trajectory"])
        reference = rule.evaluate(scenario, trajectory)
        stored = r["critic"]
        by_type.setdefault(stored["critic_type"], []).append((
            float(stored["overall_score"]), reference.overall_score,
            set(stored.get("failures", [])) == set(reference.failures),
        ))
    summary = {}
    for name, rows in by_type.items():
        observed = np.asarray([row[0] for row in rows])
        reference = np.asarray([row[1] for row in rows])
        correlation = float(np.corrcoef(observed, reference)[0, 1]) if len(rows) > 1 and np.std(observed) > 0 and np.std(reference) > 0 else 1.0
        summary[name] = {
            "count": len(rows), "mean_overall": round(float(np.mean(observed)), 3),
            "std": round(float(np.std(observed)), 3),
            "rule_mae": round(float(np.mean(np.abs(observed - reference))), 3),
            "rule_pearson": round(correlation, 4),
            "failure_exact_match": round(float(np.mean([row[2] for row in rows])), 4),
        }
    return summary
