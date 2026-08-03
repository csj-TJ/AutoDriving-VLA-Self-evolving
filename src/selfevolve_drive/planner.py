from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

import numpy as np

from .schema import Scenario, Trajectory
from .simulator import expert_target_speed


FEATURE_NAMES = ["bias", "speed_limit", "lead_gap", "lead_speed", "red", "stop_gap", "ped_gap", "curvature", "adverse"]


def scenario_features(s: Scenario) -> np.ndarray:
    return np.asarray([
        1.0, s.speed_limit, min(s.lead_distance, 60.0) / 60.0, s.lead_speed,
        float(s.traffic_light == "red"), min(s.stopline_distance, 45.0) / 45.0,
        min(s.pedestrian_distance, 60.0) / 60.0, abs(s.road_curvature),
        float(s.weather in {"rain", "fog", "night"}),
    ], dtype=float)


@dataclass
class Policy:
    name: str
    weights: list[float] | None = None
    seed: int = 42

    def target_speed(self, s: Scenario) -> float:
        if self.weights is not None:
            raw = float(scenario_features(s) @ np.asarray(self.weights))
            return min(s.speed_limit * 1.25, max(0.0, raw))
        rng = random.Random(f"{self.seed}:{s.scene_id}:{self.name}")
        expert = expert_target_speed(s)
        if self.name == "expert":
            return expert
        if self.name == "baseline":
            # Baseline intentionally misses some red-light/pedestrian/lead-vehicle cues.
            target = s.speed_limit * 0.98 + rng.gauss(0.0, 1.4)
            if s.lead_distance < 14.0 and rng.random() > 0.35:
                target = min(target, s.lead_speed)
            return max(0.0, target)
        return expert

    def plan(self, s: Scenario, horizon: int = 12, dt: float = 0.5) -> Trajectory:
        target = self.target_speed(s)
        points: list[list[float]] = []
        x, y, v = 0.0, 0.0, s.ego_speed
        route_sign = -1.0 if s.route_command == "left" else (1.0 if s.route_command == "right" else 0.0)
        for k in range(horizon):
            alpha = (k + 1) / horizon
            accel = max(-3.8, min(2.3, (target - v) * 0.45))
            v = max(0.0, v + accel * dt)
            proposed_x = x + v * dt
            if self.name != "baseline":
                # A final kinematic guard keeps the revised policy behind a moving
                # lead vehicle even when the learned target-speed model is imperfect.
                lead_x = s.lead_distance + s.lead_speed * (k + 1) * dt
                hard_limit = max(x, lead_x - 3.8)
                if proposed_x > hard_limit:
                    proposed_x = hard_limit
                    v = min(v, s.lead_speed, max(0.0, (proposed_x - x) / dt))
            x = proposed_x
            curve = s.road_curvature * (x ** 1.45) * 0.12
            route = route_sign * 1.7 * (alpha ** 2)
            y = curve + route
            points.append([round(x, 4), round(y, 4), round(v, 4)])
        rationale = f"{self.name}: target={target:.2f}m/s; light={s.traffic_light}; lead_gap={s.lead_distance:.1f}m"
        return Trajectory(points=points, target_speed=round(target, 4), rationale=rationale, policy=self.name)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "weights": self.weights, "seed": self.seed, "feature_names": FEATURE_NAMES}
