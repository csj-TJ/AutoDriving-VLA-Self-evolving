from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .critics import LLMCritic, RewardModelCritic, RuleBasedCritic
from .planner import Policy
from .reflection import reflect
from .schema import Scenario, Trajectory


CAMERA_CHANNELS = (
    "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_FRONT_LEFT",
)


def _read_table(table_root: Path, name: str) -> list[dict[str, Any]]:
    return json.loads((table_root / f"{name}.json").read_text(encoding="utf-8"))


def _yaw(rotation: list[float]) -> float:
    w, x, y, z = rotation
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _relative_xy(origin: list[float], yaw: float, point: list[float]) -> tuple[float, float]:
    dx, dy = point[0] - origin[0], point[1] - origin[1]
    forward = math.cos(yaw) * dx + math.sin(yaw) * dy
    left = -math.sin(yaw) * dx + math.cos(yaw) * dy
    return forward, -left  # Project convention: negative lateral is left.


def _distance(a: list[float], b: list[float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


class NuScenesExtractor:
    """Extract traceable lightweight training records from raw nuScenes tables."""

    def __init__(self, dataroot: Path, version: str = "v1.0-mini"):
        self.dataroot = dataroot.resolve()
        self.version = version
        self.table_root = self.dataroot / version
        if not self.table_root.is_dir():
            raise FileNotFoundError(f"nuScenes table directory not found: {self.table_root}")

        names = (
            "sample", "sample_data", "ego_pose", "scene", "log", "sensor",
            "calibrated_sensor", "sample_annotation", "instance", "category",
        )
        tables = {name: _read_table(self.table_root, name) for name in names}
        self.samples = {row["token"]: row for row in tables["sample"]}
        self.sample_order = tables["sample"]
        self.sample_data = {row["token"]: row for row in tables["sample_data"]}
        self.ego_poses = {row["token"]: row for row in tables["ego_pose"]}
        self.scenes = {row["token"]: row for row in tables["scene"]}
        self.logs = {row["token"]: row for row in tables["log"]}
        sensors = {row["token"]: row for row in tables["sensor"]}
        calibrated = {row["token"]: row for row in tables["calibrated_sensor"]}
        self.channel_by_calibrated = {
            token: sensors[row["sensor_token"]]["channel"] for token, row in calibrated.items()
        }
        self.annotations = {row["token"]: row for row in tables["sample_annotation"]}
        instances = {row["token"]: row for row in tables["instance"]}
        categories = {row["token"]: row["name"] for row in tables["category"]}
        self.category_by_instance = {
            token: categories[row["category_token"]] for token, row in instances.items()
        }
        self.data_by_sample: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        for row in tables["sample_data"]:
            if not row["is_key_frame"]:
                continue
            channel = self.channel_by_calibrated[row["calibrated_sensor_token"]]
            self.data_by_sample[row["sample_token"]][channel] = row
        self.annotations_by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in tables["sample_annotation"]:
            self.annotations_by_sample[row["sample_token"]].append(row)

        scene_names = sorted(row["name"] for row in tables["scene"])
        self.scene_split = {
            name: ("train" if index < 6 else "val" if index < 8 else "test")
            for index, name in enumerate(scene_names)
        }

    def _pose(self, sample_token: str) -> dict[str, Any]:
        data = self.data_by_sample[sample_token]
        sensor = data.get("LIDAR_TOP") or data.get("CAM_FRONT")
        if not sensor:
            raise KeyError(f"No key-frame pose for sample {sample_token}")
        return self.ego_poses[sensor["ego_pose_token"]]

    def _ego_speed(self, sample: dict[str, Any]) -> float:
        current = self._pose(sample["token"])
        neighbors = [token for token in (sample["prev"], sample["next"]) if token]
        if not neighbors:
            return 0.0
        if len(neighbors) == 2:
            before, after = self._pose(neighbors[0]), self._pose(neighbors[1])
            dt = (self.samples[neighbors[1]]["timestamp"] - self.samples[neighbors[0]]["timestamp"]) / 1e6
            return _distance(before["translation"], after["translation"]) / max(dt, 1e-3)
        other = self._pose(neighbors[0])
        dt = abs(self.samples[neighbors[0]]["timestamp"] - sample["timestamp"]) / 1e6
        return _distance(current["translation"], other["translation"]) / max(dt, 1e-3)

    def _annotation_speed(self, annotation: dict[str, Any]) -> float:
        tokens = [token for token in (annotation["prev"], annotation["next"]) if token in self.annotations]
        if not tokens:
            return 0.0
        if len(tokens) == 2:
            before, after = self.annotations[tokens[0]], self.annotations[tokens[1]]
            dt = abs(
                self.samples[after["sample_token"]]["timestamp"]
                - self.samples[before["sample_token"]]["timestamp"]
            ) / 1e6
            return _distance(before["translation"], after["translation"]) / max(dt, 1e-3)
        other = self.annotations[tokens[0]]
        dt = abs(
            self.samples[other["sample_token"]]["timestamp"]
            - self.samples[annotation["sample_token"]]["timestamp"]
        ) / 1e6
        return _distance(annotation["translation"], other["translation"]) / max(dt, 1e-3)

    def _future_trajectory(self, sample: dict[str, Any], horizon: int = 12) -> Trajectory | None:
        origin = self._pose(sample["token"])
        origin_yaw = _yaw(origin["rotation"])
        points: list[list[float]] = []
        current = sample
        previous_xy = (0.0, 0.0)
        previous_time = sample["timestamp"]
        for _ in range(horizon):
            if not current["next"]:
                return None
            current = self.samples[current["next"]]
            pose = self._pose(current["token"])
            x, y = _relative_xy(origin["translation"], origin_yaw, pose["translation"])
            dt = (current["timestamp"] - previous_time) / 1e6
            speed = math.hypot(x - previous_xy[0], y - previous_xy[1]) / max(dt, 1e-3)
            points.append([round(x, 4), round(y, 4), round(speed, 4)])
            previous_xy = (x, y)
            previous_time = current["timestamp"]
        target_speed = float(np.mean([point[2] for point in points]))
        return Trajectory(points, round(target_speed, 4), "nuScenes future ego poses at 2 Hz", "nuscenes_expert")

    def _participants(self, sample: dict[str, Any], pose: dict[str, Any]) -> tuple[float, float, float, list[str]]:
        yaw = _yaw(pose["rotation"])
        lead: tuple[float, float] | None = None
        pedestrian = 100.0
        tokens: list[str] = []
        for annotation in self.annotations_by_sample[sample["token"]]:
            tokens.append(annotation["token"])
            category = self.category_by_instance[annotation["instance_token"]]
            x, y = _relative_xy(pose["translation"], yaw, annotation["translation"])
            if x <= 0:
                continue
            if category.startswith("vehicle.") and abs(y) < 3.3:
                candidate = (x, self._annotation_speed(annotation))
                if lead is None or candidate[0] < lead[0]:
                    lead = candidate
            if category.startswith("human.pedestrian") and abs(y) < 10.0:
                pedestrian = min(pedestrian, math.hypot(x, y))
        return (
            round(lead[0], 3) if lead else 100.0,
            round(lead[1], 3) if lead else 13.9,
            round(pedestrian, 3),
            tokens,
        )

    def _source_sample(self, sample: dict[str, Any]) -> tuple[Scenario, Trajectory, dict[str, Any]] | None:
        cameras = self.data_by_sample[sample["token"]]
        if not all(channel in cameras for channel in CAMERA_CHANNELS):
            return None
        image_refs = {channel: cameras[channel]["filename"].replace("\\", "/") for channel in CAMERA_CHANNELS}
        if not all((self.dataroot / filename).is_file() for filename in image_refs.values()):
            return None
        expert = self._future_trajectory(sample)
        if expert is None:
            return None
        pose = self._pose(sample["token"])
        scene = self.scenes[sample["scene_token"]]
        log = self.logs[scene["log_token"]]
        description = scene["description"].lower()
        weather = "night" if "night" in description else "rain" if "rain" in description else "clear"
        end_x, end_y = expert.points[-1][:2]
        turn_threshold = max(2.0, abs(end_x) * .15)
        route = "left" if end_y < -turn_threshold else "right" if end_y > turn_threshold else "straight"
        denominator = max(end_x * end_x + end_y * end_y, 1.0)
        curvature = max(-.13, min(.13, 2.0 * end_y / denominator))
        lead_distance, lead_speed, pedestrian_distance, annotation_tokens = self._participants(sample, pose)
        scenario = Scenario(
            scene_id=f"nuscenes-{sample['token']}",
            ego_speed=round(self._ego_speed(sample), 3),
            speed_limit=13.9,
            lead_distance=lead_distance,
            lead_speed=lead_speed,
            traffic_light="green",
            stopline_distance=45.0,
            pedestrian_distance=pedestrian_distance,
            road_curvature=round(curvature, 5),
            route_command=route,
            weather=weather,
            unseen=self.scene_split[scene["name"]] == "test",
        )
        source = {
            "dataset": "nuScenes",
            "version": self.version,
            "sample_token": sample["token"],
            "scene_token": sample["scene_token"],
            "scene_name": scene["name"],
            "scene_description": scene["description"],
            "timestamp": sample["timestamp"],
            "log_token": scene["log_token"],
            "location": log["location"],
            "ego_pose_token": cameras["LIDAR_TOP"]["ego_pose_token"],
            "camera_sample_data_tokens": {channel: cameras[channel]["token"] for channel in CAMERA_CHANNELS},
            "image_refs": image_refs,
            "annotation_tokens": annotation_tokens,
            "expert_trajectory_source": "future ego poses from sample.next chain",
            "field_notes": {
                "speed_limit": "nuScenes has no speed-limit table; fixed urban prior 13.9 m/s",
                "traffic_light": "nuScenes has no signal-phase annotation; neutral green value",
                "stopline_distance": "map expansion not required for mini pilot; neutral 45 m value",
            },
        }
        return scenario, expert, source

    def extract(
        self, config: dict[str, Any], rollout_variants: int = 18
    ) -> tuple[list[dict[str, Any]], dict[str, Any], RewardModelCritic]:
        base: list[tuple[Scenario, Trajectory, dict[str, Any]]] = []
        for sample in self.sample_order:
            extracted = self._source_sample(sample)
            if extracted:
                base.append(extracted)
        if not base:
            raise RuntimeError("No complete nuScenes samples with a 6-second future horizon were found.")

        rule = RuleBasedCritic(config["critic_weights"])
        raw: list[tuple[Scenario, Trajectory, Trajectory, dict[str, Any], Any]] = []
        for scenario, expert, source in base:
            for variant in range(rollout_variants):
                variant_scenario = replace(scenario, scene_id=f"{scenario.scene_id}-{variant:02d}")
                rollout = Policy("baseline", seed=config["seed"] + variant).plan(
                    variant_scenario, config["trajectory_horizon"], config["dt"]
                )
                raw.append((variant_scenario, rollout, expert, {**source, "rollout_variant": variant}, rule.evaluate(variant_scenario, rollout)))

        reward = RewardModelCritic.fit(
            [(scenario, rollout, result) for scenario, rollout, _, _, result in raw[: max(200, len(raw) // 2)]],
            config["ridge_lambda"],
        )
        llm = LLMCritic(rule, config["llm"])
        records: list[dict[str, Any]] = []
        for index, (scenario, rollout, expert, source, rule_result) in enumerate(raw):
            critic = rule_result
            if index % 10 == 1:
                critic = llm.evaluate(scenario, rollout)
            elif index % 10 == 2:
                critic = reward.evaluate(scenario, rollout)
                critic.failures = rule_result.failures
                critic.evidence = rule_result.evidence + critic.evidence
            reflection = reflect(scenario, rollout, critic)
            quality = round(.65 * reflection.confidence + .35 * min(1.0, len(critic.evidence) / 2.0), 3)
            scene_name = source["scene_name"]
            records.append({
                "sample_id": f"nuscenes-{source['sample_token']}-{source['rollout_variant']:02d}",
                "source": source,
                "image_refs": [source["image_refs"][channel] for channel in CAMERA_CHANNELS],
                "scenario": scenario.to_dict(),
                "expert_trajectory": expert.to_dict(),
                "trajectory": rollout.to_dict(),
                "critic": critic.to_dict(),
                "reflection": reflection.to_dict(),
                "quality": quality,
                "accepted": quality >= config["quality_threshold"],
                "split": self.scene_split[scene_name],
            })
        manifest = {
            "dataset": "nuScenes",
            "version": self.version,
            "dataroot": "data/nuscenes",
            "raw_keyframes": len(self.sample_order),
            "keyframes_with_6s_future_and_six_cameras": len(base),
            "rollout_variants_per_keyframe": rollout_variants,
            "records": len(records),
            "splits": {name: sum(row["split"] == name for row in records) for name in ("train", "val", "test")},
            "accepted": sum(row["accepted"] for row in records),
            "camera_channels": list(CAMERA_CHANNELS),
            "all_image_refs_verified": True,
            "sample_token_traceability": True,
            "expert_trajectory": "12 future ego poses at approximately 2 Hz",
            "source_archive": "data/nuscenes/v1.0-mini.tgz",
        }
        return records, manifest, reward


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
