from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from selfevolve_drive.evaluation import critic_agreement, evaluate_policy
from selfevolve_drive.pipeline import load_config, read_jsonl
from selfevolve_drive.planner import Policy
from selfevolve_drive.schema import Scenario, Trajectory
from selfevolve_drive.training import record_target_speed


def load_policy(path: Path) -> Policy:
    obj = json.loads(path.read_text(encoding="utf-8"))
    return Policy(obj["name"], obj["weights"], obj.get("seed", 42))


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate baseline/SFT/reflection policies and Critic consistency")
    p.add_argument("--config", default=str(ROOT / "configs" / "default.json"))
    p.add_argument("--data", default=str(ROOT / "data" / "reflection_dataset.jsonl"))
    p.add_argument("--output", default=str(ROOT / "outputs" / "metrics.json"))
    p.add_argument("--model-dir", default=str(ROOT / "models"))
    args = p.parse_args()
    cfg, records = load_config(args.config), read_jsonl(args.data)
    test_records = [record for record in records if record["split"] == "test"]
    test_scenes = [Scenario(**r["scenario"]) for r in test_records]
    test_targets = [record_target_speed(record) for record in test_records]
    test_experts = [Trajectory(**record["expert_trajectory"]) for record in test_records]
    sample_tokens = [record.get("source", {}).get("sample_token", record["sample_id"]) for record in test_records]
    policies = [Policy("baseline", seed=cfg["seed"])]
    for name in ("sft", "reflection_sft", "reflection_dpo"):
        path = Path(args.model_dir) / f"{name}.json"
        if path.exists(): policies.append(load_policy(path))
    source_dataset = records[0].get("source", {}).get("dataset") if records else None
    scope = (
        "nuScenes-derived lightweight open-loop experiment; not a CARLA or on-road closed-loop claim."
        if source_dataset == "nuScenes"
        else "Lightweight simulator-backed open-loop proxy metrics; not CARLA closed-loop claims."
    )
    result = {
        "benchmark": {
            "name": "nuScenes-mini scene-disjoint open-loop planning",
            "protocol": "12 waypoints at 0.5 s; report L2 at 1/2/3 s, risk rates, comfort and clustered 95% CI",
            "reference_protocols": ["ST-P3 open-loop", "UniAD open-loop"],
            "literature_reference": {
                "source": "OpenDriveVLA arXiv:2503.23463v2 Table I",
                "OpenDriveVLA-0.5B_ST-P3": {"l2_1s": .15, "l2_2s": .32, "l2_3s": .57, "avg_l2": .35, "avg_collision_percent": .09},
                "OpenDriveVLA-0.5B_UniAD": {"l2_1s": .21, "l2_2s": .60, "l2_3s": 1.22, "avg_l2": .68, "avg_collision_percent": .26},
                "UniAD_UniAD_protocol": {"l2_1s": .48, "l2_2s": .96, "l2_3s": 1.65, "avg_l2": 1.03, "avg_collision_percent": .31},
            },
        },
        "dataset": {
            "name": source_dataset or "lightweight_proxy", "total": len(records),
            "test_rollouts": len(test_scenes), "test_unique_keyframes": len(set(sample_tokens)),
            "accepted": sum(r["accepted"] for r in records),
        },
        "policies": [evaluate_policy(
            x, test_scenes, cfg["critic_weights"], test_targets, test_experts, sample_tokens
        ) for x in policies],
        "critic_summary": critic_agreement(records), "scope_note": scope,
    }
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
