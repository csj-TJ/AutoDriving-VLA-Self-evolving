from __future__ import annotations

from dataclasses import dataclass
import math
import random
from .schema import Scenario


WEATHERS = ("clear", "rain", "fog", "night")
COMMANDS = ("straight", "left", "right")
PEDESTRIAN_SAFETY_RADIUS_M = 2.0


@dataclass(frozen=True)
class PedestrianConflict:
    relevant: bool
    minimum_separation_m: float | None
    conflict_time_s: float | None
    conflict_x_m: float | None
    safe_target_speed: float | None
    source: str


def route_lateral_position(s: Scenario, x: float, progress: float) -> float:
    route_sign = -1.0 if s.route_command == "left" else (1.0 if s.route_command == "right" else 0.0)
    curve = s.road_curvature * (max(0.0, x) ** 1.45) * 0.12
    return curve + route_sign * 1.7 * (max(0.0, min(1.0, progress)) ** 2)


def _pedestrian_points(s: Scenario) -> list[tuple[float, float, float]]:
    points: list[tuple[float, float, float]] = []
    for point in s.pedestrian_track:
        if len(point) >= 3:
            values = tuple(float(value) for value in point[:3])
            if all(math.isfinite(value) for value in values):
                points.append(values)
    if not points and s.pedestrian_x is not None and s.pedestrian_y is not None:
        points.append((0.0, float(s.pedestrian_x), float(s.pedestrian_y)))
    return sorted(points)


def _position_at(points: list[tuple[float, float, float]], t: float) -> tuple[float, float]:
    if len(points) == 1 or t <= points[0][0]:
        return points[0][1], points[0][2]
    if t >= points[-1][0]:
        return points[-1][1], points[-1][2]
    for left, right in zip(points, points[1:]):
        if left[0] <= t <= right[0]:
            span = max(1e-6, right[0] - left[0])
            ratio = (t - left[0]) / span
            return (left[1] + (right[1] - left[1]) * ratio,
                    left[2] + (right[2] - left[2]) * ratio)
    return points[-1][1], points[-1][2]


def _ego_points(
    s: Scenario,
    target_speed: float,
    horizon: int = 12,
    dt: float = 0.5,
) -> list[tuple[float, float, float]]:
    points = [(0.0, 0.0, 0.0)]
    x, v = 0.0, s.ego_speed
    for index in range(horizon):
        accel = max(-3.8, min(2.3, (target_speed - v) * 0.45))
        v = max(0.0, v + accel * dt)
        x += v * dt
        progress = (index + 1) / horizon
        points.append(((index + 1) * dt, x, route_lateral_position(s, x, progress)))
    return points


def _ego_position_at(points: list[tuple[float, float, float]], t: float) -> tuple[float, float]:
    if t <= points[0][0]:
        return points[0][1], points[0][2]
    if t >= points[-1][0]:
        return points[-1][1], points[-1][2]
    for left, right in zip(points, points[1:]):
        if left[0] <= t <= right[0]:
            span = max(1e-6, right[0] - left[0])
            ratio = (t - left[0]) / span
            return (left[1] + (right[1] - left[1]) * ratio,
                    left[2] + (right[2] - left[2]) * ratio)
    return points[-1][1], points[-1][2]


def assess_pedestrian_conflict(
    s: Scenario,
    trajectory_points: list[list[float]] | None = None,
    target_speed: float | None = None,
    dt: float = 0.5,
) -> PedestrianConflict:
    """Estimate time-aligned ego/pedestrian clearance with a deterministic CPU model."""
    pedestrian = _pedestrian_points(s)
    if not pedestrian:
        relevant = s.pedestrian_distance < 18.0
        safe_speed = max(0.0, (s.pedestrian_distance - 3.0) / 2.5) if relevant else None
        return PedestrianConflict(
            relevant, s.pedestrian_distance if relevant else None,
            s.pedestrian_distance / max(s.ego_speed, 0.5) if relevant else None,
            s.pedestrian_distance if relevant else None, safe_speed, "legacy_distance",
        )

    if trajectory_points is None:
        ego = _ego_points(s, s.speed_limit if target_speed is None else target_speed, dt=dt)
    else:
        ego = [(0.0, 0.0, 0.0)] + [
            ((index + 1) * dt, float(point[0]), float(point[1]))
            for index, point in enumerate(trajectory_points)
        ]
    horizon = ego[-1][0]
    sample_count = max(1, int(round(horizon / 0.1)))
    best: tuple[float, float, float] | None = None
    for index in range(sample_count + 1):
        t = min(horizon, index * 0.1)
        ego_x, ego_y = _ego_position_at(ego, t)
        ped_x, ped_y = _position_at(pedestrian, t)
        separation = math.hypot(ego_x - ped_x, ego_y - ped_y)
        if best is None or separation < best[0]:
            best = (separation, t, ped_x)
    assert best is not None
    relevant = best[0] < PEDESTRIAN_SAFETY_RADIUS_M and best[2] >= -1.0
    # A predicted path crossing requires a yield/stop target. The longitudinal
    # guard in the planner remains a final containment layer for short gaps.
    safe_speed = 0.0 if relevant else None
    return PedestrianConflict(
        relevant, round(best[0], 3), round(best[1], 3), round(best[2], 3),
        round(safe_speed, 3) if safe_speed is not None else None, "spatiotemporal_track",
    )


def pedestrian_feature_distance(s: Scenario) -> float:
    """Map explicit 2-D geometry onto the legacy scalar feature without false side-lane proximity."""
    pedestrian = _pedestrian_points(s)
    if not pedestrian:
        return s.pedestrian_distance
    crossing_x = [
        x for t, x, y in pedestrian
        if x >= -1.0 and abs(y - route_lateral_position(s, x, min(1.0, t / 6.0)))
        < PEDESTRIAN_SAFETY_RADIUS_M
    ]
    return min(crossing_x) if crossing_x else 100.0


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
    pedestrian = assess_pedestrian_conflict(s, target_speed=target)
    if pedestrian.relevant and pedestrian.safe_target_speed is not None:
        target = min(target, pedestrian.safe_target_speed)
    target *= max(0.55, 1.0 - abs(s.road_curvature) * 3.0)
    return max(0.0, target)

