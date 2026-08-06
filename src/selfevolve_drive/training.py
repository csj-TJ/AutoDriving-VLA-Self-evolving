from __future__ import annotations

from typing import Iterable
import numpy as np

from .planner import Policy, scenario_features
from .schema import Scenario
from .simulator import expert_target_speed


def record_target_speed(record: dict) -> float:
    expert = record.get("expert_trajectory")
    if expert and "target_speed" in expert:
        return float(expert["target_speed"])
    return expert_target_speed(Scenario(**record["scenario"]))


def fit_sft_records(records: list[dict], name: str = "sft", ridge: float = 0.1) -> Policy:
    scenes = [Scenario(**record["scenario"]) for record in records]
    x = np.vstack([scenario_features(scene) for scene in scenes])
    y = np.asarray([record_target_speed(record) for record in records])
    weights = np.linalg.solve(x.T @ x + ridge * np.eye(x.shape[1]), x.T @ y)
    return Policy(name=name, weights=weights.tolist())


def fit_sft(scenes: Iterable[Scenario], name: str = "sft", ridge: float = 0.1) -> Policy:
    scenes = list(scenes)
    x = np.vstack([scenario_features(s) for s in scenes])
    y = np.asarray([expert_target_speed(s) for s in scenes])
    w = np.linalg.solve(x.T @ x + ridge * np.eye(x.shape[1]), x.T @ y)
    return Policy(name=name, weights=w.tolist())


def fit_reflection_sft(records: list[dict], ridge: float = 0.1) -> Policy:
    # Failures are deliberately upweighted: reflection turns rare mistakes into training signal.
    scenes: list[Scenario] = []
    weights: list[float] = []
    for r in records:
        s = Scenario(**r["scenario"])
        scenes.append(s)
        failures = r["critic"]["failures"]
        weights.append(1.0 + min(4.0, len(failures) * 1.25))
    x = np.vstack([scenario_features(s) for s in scenes])
    y = np.asarray([record_target_speed(record) for record in records])
    sw = np.sqrt(np.asarray(weights))[:, None]
    xw, yw = x * sw, y * sw[:, 0]
    w = np.linalg.solve(xw.T @ xw + ridge * np.eye(x.shape[1]), xw.T @ yw)
    return Policy(name="reflection_sft", weights=w.tolist())


def fit_dpo(records: list[dict], reference: Policy, beta: float = 0.15, epochs: int = 4) -> Policy:
    w = np.asarray(reference.weights, dtype=float).copy()
    for _ in range(epochs):
        for r in records:
            if r["reflection"]["verdict"] != "revise":
                continue
            s = Scenario(**r["scenario"])
            x = scenario_features(s)
            chosen = record_target_speed(r)
            rejected = float(r["trajectory"]["target_speed"])
            pred = float(x @ w)
            grad = 2.0 * ((pred - chosen) - 0.25 * (pred - rejected)) * x
            w -= beta * grad / (1.0 + float(x @ x))
    return Policy(name="reflection_dpo", weights=w.tolist())
