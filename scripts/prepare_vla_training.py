from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from selfevolve_drive.pipeline import read_jsonl
from selfevolve_drive.vla_training_data import export_training_sets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export OpenDriveVLA SFT, Reflection-SFT and DPO datasets")
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "reflection_dataset.jsonl")
    parser.add_argument("--models", type=Path, default=ROOT / "models")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "opendrivevla_training")
    args = parser.parse_args()
    print(json.dumps(export_training_sets(read_jsonl(args.data), args.models, args.output), ensure_ascii=False, indent=2))
