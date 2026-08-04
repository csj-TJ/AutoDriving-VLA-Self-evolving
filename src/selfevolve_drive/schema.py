from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Scenario:
    scene_id: str
    ego_speed: float
    speed_limit: float
    lead_distance: float
    lead_speed: float
    traffic_light: str
    stopline_distance: float
    pedestrian_distance: float
    road_curvature: float
    route_command: str
    weather: str
    unseen: bool = False
    pedestrian_x: float | None = None
    pedestrian_y: float | None = None
    pedestrian_track: list[list[float]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Trajectory:
    points: list[list[float]]
    target_speed: float
    rationale: str
    policy: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CriticResult:
    critic_type: str
    safety_score: float
    rule_score: float
    comfort_score: float
    overall_score: float
    failures: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Reflection:
    verdict: str
    root_causes: list[str]
    evidence: list[str]
    corrective_strategy: list[str]
    counterfactual_action: str
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

