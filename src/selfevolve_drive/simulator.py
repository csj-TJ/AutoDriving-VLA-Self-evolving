from __future__ import annotations

import random
from .schema import Scenario


WEATHERS = ("clear", "rain", "fog", "night")
COMMANDS = ("straight", "left", "right")


def generate_scenarios(count: int, seed: int = 42, unseen_fraction: float = 0.2) -> list[Scenario]:
    rng = random.Random(seed)
    scenes: list[Scenario] = []
    for i in range(count):
        unseen = i >= int(count * (1.0 - unseen_fraction))
        weather_pool = WEATHERS if unseen else WEATHERS[:3]
        speed_limit = rng.choice((8.0, 10.0, 13.9, 16.7))
        red = rng.random() < (0.28 if unseen else 0.20)
        ped = rng.uniform(4.0, 45.0) if rng.random() < 0.22 else 100.0
        curvature = rng.uniform(-0.13, 0.13) if unseen else rng.uniform(-0.08, 0.08)
        scenes.append(Scenario(
            scene_id=f"scene-{i:06d}",
            ego_speed=max(0.0, rng.gauss(speed_limit * 0.82, 2.2)),
            speed_limit=speed_limit,
            lead_distance=rng.uniform(5.0, 65.0),
            lead_speed=max(0.0, rng.gauss(speed_limit * 0.72, 2.8)),
            traffic_light="red" if red else "green",
            stopline_distance=rng.uniform(5.0, 45.0),
            pedestrian_distance=ped,
            road_curvature=curvature,
            route_command=rng.choice(COMMANDS),
            weather=rng.choice(weather_pool),
            unseen=unseen,
        ))
    return scenes


def expert_target_speed(s: Scenario) -> float:
    target = s.speed_limit * (0.78 if s.weather in {"rain", "fog", "night"} else 0.92)
    if s.lead_distance < 25.0:
        target = min(target, max(0.0, s.lead_speed - 0.5))
    if s.traffic_light == "red":
        target = min(target, max(0.0, (s.stopline_distance - 2.5) / 3.0))
    if s.pedestrian_distance < 18.0:
        target = min(target, max(0.0, (s.pedestrian_distance - 3.0) / 2.5))
    target *= max(0.55, 1.0 - abs(s.road_curvature) * 3.0)
    return max(0.0, target)

