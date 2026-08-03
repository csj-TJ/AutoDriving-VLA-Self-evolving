from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(script: str, extra: list[str] | None = None) -> None:
    subprocess.run([sys.executable, str(ROOT / "scripts" / script), *(extra or [])], check=True, cwd=ROOT)


def main() -> None:
    p = argparse.ArgumentParser(description="One-command reproducibility pipeline")
    p.add_argument("--samples", type=int, default=5000)
    args = p.parse_args()
    run("generate_data.py", ["--samples", str(args.samples)])
    run("train.py")
    run("prepare_vla_training.py")
    run("evaluate.py")
    print("Pipeline complete. Open outputs/metrics.json or run: python scripts/run_demo.py")


if __name__ == "__main__":
    main()
