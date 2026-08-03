from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from selfevolve_drive.base_models import parse_opendrivevla_trajectory


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Import official OpenDriveVLA plan_conv output for the lightweight demo")
    parser.add_argument("--plan-conv", type=Path, required=True, help="Official plan_conv.json JSONL")
    parser.add_argument("--scenario-manifest", type=Path, required=True, help="JSONL with id and demo Scenario fields")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "opendrivevla" / "demo_cache.jsonl")
    args = parser.parse_args()

    scenarios = {str(item["id"]): item for item in read_jsonl(args.scenario_manifest)}
    imported: list[dict] = []
    skipped = 0
    for result in read_jsonl(args.plan_conv):
        source_id = str(result.get("id", ""))
        scenario = scenarios.get(source_id) or scenarios.get(source_id.removesuffix("_trajectory"))
        if not scenario:
            skipped += 1
            continue
        answer = result.get("answer", "")
        if isinstance(answer, list):
            answer = answer[0] if answer else ""
        try:
            trajectory = parse_opendrivevla_trajectory(str(answer), "opendrivevla_baseline")
        except ValueError:
            skipped += 1
            continue
        scene_id = str(scenario.get("scene_id") or scenario["id"])
        imported.append({
            "scene_id": scene_id,
            "source_id": source_id,
            "source_model": "OpenDriveVLA-0.5B",
            "source_type": "official_gpu_inference_cache",
            "scenario": {k: v for k, v in scenario.items() if k != "id"},
            "trajectory": trajectory.to_dict(),
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in imported),
        encoding="utf-8",
    )
    print(f"imported={len(imported)} skipped={skipped} output={args.output}")


if __name__ == "__main__":
    main()
