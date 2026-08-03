from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from selfevolve_drive.web_demo import serve


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Launch the local interactive demo")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()
    serve(args.host, args.port)

