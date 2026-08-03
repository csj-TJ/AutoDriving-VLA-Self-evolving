from __future__ import annotations

import json
from pathlib import Path

from .planner import Policy
from .schema import Scenario, Trajectory


def trajectory_text(trajectory: Trajectory) -> str:
    pairs = ",".join(f"({p[0]:.3f},{p[1]:.3f})" for p in trajectory.points[:6])
    return f"<traj_start>[{pairs}]<traj_end>"


def scenario_prompt(scene: Scenario) -> str:
    return (
        "You are Open-DriveVLA. Plan a safe 3-second trajectory. "
        f"Ego speed={scene.ego_speed:.2f}m/s; speed limit={scene.speed_limit:.2f}m/s; "
        f"lead distance={scene.lead_distance:.2f}m; lead speed={scene.lead_speed:.2f}m/s; "
        f"traffic light={scene.traffic_light}; stopline={scene.stopline_distance:.2f}m; "
        f"pedestrian distance={scene.pedestrian_distance:.2f}m; curvature={scene.road_curvature:.4f}; "
        f"command={scene.route_command}; weather={scene.weather}."
    )


def export_training_sets(records: list[dict], model_dir: Path, output_dir: Path) -> dict:
    policies = {}
    for name in ("sft", "reflection_sft"):
        obj = json.loads((model_dir / f"{name}.json").read_text(encoding="utf-8"))
        policies[name] = Policy(obj["name"], obj["weights"], obj.get("seed", 42))

    sft, reflection_sft, preferences = [], [], []
    for record in records:
        if record["split"] != "train" or not record["accepted"]:
            continue
        scene = Scenario(**record["scenario"])
        ordinary = policies["sft"].plan(scene, horizon=6)
        corrected = policies["reflection_sft"].plan(scene, horizon=6)
        prompt = scenario_prompt(scene)
        common = {
            "id": record["sample_id"],
            "scene_id": scene.scene_id,
            "base_model": "OpenDriveVLA-0.5B",
            "source_type": "nuscenes_v1.0-mini",
            "sample_token": record.get("sample_token", record.get("source", {}).get("sample_token")),
            "image_refs": record.get("image_refs", []),
            "annotation_tokens": record.get("source", {}).get("annotation_tokens", []),
            "source": record.get("source", {}),
        }
        sft.append({**common, "prompt": prompt, "response": trajectory_text(ordinary)})
        reflection = record["reflection"]
        reflection_prompt = (
            prompt + "\nInitial trajectory: " + trajectory_text(Trajectory(**record["trajectory"])) +
            "\nCritic: " + json.dumps(record["critic"], ensure_ascii=False) +
            "\nDriving Reflection: " + json.dumps(reflection, ensure_ascii=False) +
            "\nGenerate the corrected trajectory only."
        )
        reflection_sft.append({**common, "prompt": reflection_prompt, "response": trajectory_text(corrected)})
        if reflection["verdict"] == "revise":
            preferences.append({
                **common,
                "prompt": reflection_prompt,
                "chosen": trajectory_text(corrected),
                "rejected": trajectory_text(Trajectory(**record["trajectory"])),
            })

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "sft": (output_dir / "sft.jsonl", sft),
        "reflection_sft": (output_dir / "reflection_sft.jsonl", reflection_sft),
        "dpo": (output_dir / "dpo.jsonl", preferences),
    }
    for _, (path, rows) in outputs.items():
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "base_model": "OpenDriveVLA-0.5B",
        "initialization": "references/models/OpenDriveVLA-0.5B",
        "source_scope": "nuScenes v1.0-mini keyframes with six-camera references, annotation provenance and future ego-pose supervision",
        "counts": {key: len(rows) for key, (_, rows) in outputs.items()},
        "files": {key: f"data/opendrivevla_training/{path.name}" for key, (path, _) in outputs.items()},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
