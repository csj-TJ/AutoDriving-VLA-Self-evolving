from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from selfevolve_drive.base_models import audit_opendrivevla


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit local OpenDriveVLA assets and runtime")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args()
    result = audit_opendrivevla(ROOT)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("OpenDriveVLA 本地查验")
        print(f"- checkpoint: {'完整' if result['model_complete'] else '缺失/损坏'}")
        print(f"- official code: {'完整' if result['code_complete'] else '不完整'}")
        print(f"- safetensors header: {'通过' if result['safetensors_header_ok'] else '失败'}")
        print(f"- architecture: {result['architecture']}")
        print(f"- current Python runtime: {'可运行官方推理' if result['runtime_ready'] else '未安装官方 GPU 依赖'}")
        print(f"- cache: {'可用' if result['cache_available'] else '尚无真实推理缓存'}")
        if result["missing_model_files"]:
            print("- missing model files: " + ", ".join(result["missing_model_files"]))
        if result["missing_code_components"]:
            print("- missing code components: " + ", ".join(result["missing_code_components"]))

