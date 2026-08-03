from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from selfevolve_drive.nuscenes_data import NuScenesExtractor, sha256_file
from selfevolve_drive.pipeline import load_config, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract traceable nuScenes Reflection training records.")
    parser.add_argument("--dataroot", default=str(ROOT / "data" / "nuscenes"))
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.json"))
    parser.add_argument("--output", default=str(ROOT / "data" / "reflection_dataset.jsonl"))
    parser.add_argument("--manifest", default=str(ROOT / "data" / "nuscenes_manifest.json"))
    parser.add_argument("--reward-model", default=str(ROOT / "models" / "reward_critic.json"))
    parser.add_argument("--rollout-variants", type=int, default=18)
    args = parser.parse_args()

    dataroot = Path(args.dataroot)
    records, manifest, reward = NuScenesExtractor(dataroot, args.version).extract(
        load_config(args.config), args.rollout_variants
    )
    archive = dataroot / f"{args.version}.tgz"
    if archive.is_file():
        manifest["source_archive_bytes"] = archive.stat().st_size
        manifest["source_archive_sha256"] = sha256_file(archive)
    write_jsonl(args.output, records)
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    reward_path = Path(args.reward_model)
    reward_path.parent.mkdir(parents=True, exist_ok=True)
    reward_path.write_text(json.dumps(reward.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
