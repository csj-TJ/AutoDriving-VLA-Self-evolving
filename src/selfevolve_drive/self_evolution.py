from __future__ import annotations

from collections import Counter

import numpy as np

from .critics import RuleBasedCritic
from .planner import Policy, scenario_features
from .reflection import reflect
from .schema import Scenario
from .simulator import expert_target_speed


def _weighted_fit(scenes: list[Scenario], sample_weights: list[float], ridge: float, name: str) -> Policy:
    x = np.vstack([scenario_features(scene) for scene in scenes])
    y = np.asarray([expert_target_speed(scene) for scene in scenes])
    scale = np.sqrt(np.asarray(sample_weights))[:, None]
    xw, yw = x * scale, y * scale[:, 0]
    weights = np.linalg.solve(xw.T @ xw + ridge * np.eye(x.shape[1]), xw.T @ yw)
    return Policy(name=name, weights=weights.tolist())


def run_self_evolution(
    records: list[dict],
    critic_weights: dict[str, float],
    ridge: float = 0.1,
    rounds: int = 3,
) -> tuple[list[Policy], list[dict], list[dict]]:
    """Run local failure replay: rollout -> critic -> reflection -> weighted refit."""
    train_records = [r for r in records if r["split"] == "train" and r["accepted"]]
    scenes = [Scenario(**r["scenario"]) for r in train_records]
    critic = RuleBasedCritic(critic_weights)
    policy = _weighted_fit(scenes, [1.0] * len(scenes), ridge, "evolution_round_0")
    policies: list[Policy] = []
    history: list[dict] = []
    memory: list[dict] = []

    for round_id in range(1, rounds + 1):
        sample_weights: list[float] = []
        failures = Counter()
        scores: list[float] = []
        revised = 0
        for source, scene in zip(train_records, scenes):
            trajectory = policy.plan(scene)
            result = critic.evaluate(scene, trajectory)
            reflection = reflect(scene, trajectory, result)
            failures.update(result.failures)
            scores.append(result.overall_score)
            if reflection.verdict == "revise":
                revised += 1
                memory.append({
                    "round": round_id,
                    "sample_id": source["sample_id"],
                    "scenario": scene.to_dict(),
                    "trajectory": trajectory.to_dict(),
                    "critic": result.to_dict(),
                    "reflection": reflection.to_dict(),
                    "target_speed": expert_target_speed(scene),
                    "source_type": "synthetic_proxy_self_evolution",
                })
            severity = len(result.failures)
            sample_weights.append(1.0 + severity * (1.0 + 0.4 * round_id) + (1.0 - result.overall_score / 100.0))

        history.append({
            "round": round_id,
            "policy": policy.name,
            "samples": len(scenes),
            "revised_samples": revised,
            "mean_overall": round(float(np.mean(scores)), 3),
            "failure_events": int(sum(failures.values())),
            "failure_breakdown": dict(failures),
            "memory_size": len(memory),
        })
        policy = _weighted_fit(scenes, sample_weights, ridge, f"evolution_round_{round_id}")
        policies.append(policy)

    return policies, history, memory
