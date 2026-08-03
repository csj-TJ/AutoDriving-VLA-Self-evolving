from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from selfevolve_drive.pipeline import build_records, load_config, write_jsonl


def main() -> None:
    p = argparse.ArgumentParser(description="Generate Trajectory->Critic->Reflection JSONL data")
    p.add_argument("--config", default=str(ROOT / "configs" / "default.json"))
    p.add_argument("--output", default=str(ROOT / "data" / "reflection_dataset.jsonl"))
    p.add_argument("--samples", type=int, default=None)
    args = p.parse_args()
    cfg = load_config(args.config)
    if args.samples is not None:
        cfg["num_samples"] = args.samples
    records, reward = build_records(cfg)
    write_jsonl(args.output, records)
    model_path = ROOT / "models" / "reward_critic.json"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text(json.dumps(reward.to_dict(), indent=2), encoding="utf-8")
    accepted = sum(r["accepted"] for r in records)
    print(json.dumps({"output": args.output, "samples": len(records), "accepted": accepted, "acceptance_rate": round(accepted / len(records), 4)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

