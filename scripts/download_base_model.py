from __future__ import annotations

import argparse
import getpass
import os
from pathlib import Path


REPO_ID = "OpenDriveVLA/OpenDriveVLA-0.5B"
MODEL_URL = "https://huggingface.co/OpenDriveVLA/OpenDriveVLA-0.5B"
ROOT = Path(__file__).resolve().parents[1]
INFERENCE_FILES = [
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "added_tokens.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "merges.txt",
    "vocab.json",
]


def normalize_token(token: str) -> str:
    token = token.strip().strip('"').strip("'").strip()
    if token.lower().startswith("bearer "):
        token = token[7:].strip()
    return token


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the required OpenDriveVLA-0.5B checkpoint")
    parser.add_argument("--output", type=Path, default=ROOT / "references" / "models" / "OpenDriveVLA-0.5B")
    parser.add_argument("--token", default=os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN"))
    args = parser.parse_args()
    if not args.token:
        print("请确认已在以下页面获得模型访问权限：")
        print(MODEL_URL)
        args.token = getpass.getpass("请粘贴 Hugging Face 只读 Token（输入不会显示）：").strip()
    args.token = normalize_token(args.token or "")
    if not args.token:
        raise SystemExit("未输入 Token，下载已取消。")
    try:
        from huggingface_hub import HfApi, snapshot_download
        from huggingface_hub.errors import GatedRepoError, HfHubHTTPError
    except ImportError as exc:
        raise SystemExit("请先安装：python -m pip install -r requirements-model.txt") from exc

    api = HfApi(token=args.token)
    try:
        account = api.whoami()
        api.model_info(REPO_ID, token=args.token)
    except (GatedRepoError, HfHubHTTPError) as exc:
        raise SystemExit(
            "Token 验证失败或该 Token 无权访问 OpenDriveVLA-0.5B。\n"
            "请依次检查：\n"
            "1. 浏览器获批模型访问权的账号与创建 Token 的账号必须相同；\n"
            "2. 创建 Read Token，或给 Fine-grained Token 显式添加该模型仓库的读取权限；\n"
            "3. 重新运行脚本并粘贴以 hf_ 开头的 Token，不要粘贴网页密码。\n"
            f"模型页面：{MODEL_URL}\n"
            f"Hugging Face 返回：{exc}"
        ) from exc

    username = account.get("name") or account.get("fullname") or "已认证用户"
    print(f"认证成功：{username}")
    print(f"将下载 {len(INFERENCE_FILES)} 个推理必需文件到：{args.output.resolve()}")
    args.output.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=REPO_ID,
        local_dir=args.output,
        token=args.token,
        allow_patterns=INFERENCE_FILES,
        max_workers=4,
    )
    print(f"模型已下载到：{args.output.resolve()}")


if __name__ == "__main__":
    main()
