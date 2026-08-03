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
from selfevolve_drive.schema import Scenario


def load_policy(path: Path) -> Policy:
    obj = json.loads(path.read_text(encoding="utf-8"))
    return Policy(obj["name"], obj["weights"], obj.get("seed", 42))


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate baseline/SFT/reflection policies and Critic consistency")
    p.add_argument("--config", default=str(ROOT / "configs" / "default.json"))
    p.add_argument("--data", default=str(ROOT / "data" / "reflection_dataset.jsonl"))
    p.add_argument("--output", default=str(ROOT / "outputs" / "metrics.json"))
    args = p.parse_args()
    cfg, records = load_config(args.config), read_jsonl(args.data)
    test_scenes = [Scenario(**r["scenario"]) for r in records if r["split"] == "test"]
    policies = [Policy("baseline", seed=cfg["seed"])]
    for name in ("sft", "reflection_sft", "reflection_dpo"):
        path = ROOT / "models" / f"{name}.json"
        if path.exists(): policies.append(load_policy(path))
    result = {"dataset": {"total": len(records), "test": len(test_scenes), "accepted": sum(r["accepted"] for r in records)}, "policies": [evaluate_policy(x, test_scenes, cfg["critic_weights"]) for x in policies], "critic_summary": critic_agreement(records), "scope_note": "Lightweight synthetic open-loop proxy metrics; not CARLA closed-loop claims."}
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

