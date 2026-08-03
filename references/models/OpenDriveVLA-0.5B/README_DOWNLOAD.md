# OpenDriveVLA-0.5B 本地权重目录

模型权重不提交到 GitHub。协作者需先在 Hugging Face 接受模型访问条件，然后将仓库完整下载到本目录：

https://huggingface.co/OpenDriveVLA/OpenDriveVLA-0.5B

项目提供交互式下载脚本：

```powershell
python scripts/download_base_model.py
```

下载完成后验证：

```powershell
python scripts/audit_opendrivevla.py
```

至少应包含 `model.safetensors`、`config.json`、tokenizer 配置、`merges.txt` 和 `vocab.json`。当前官方权重约 1.47 GB，请勿绕过 `.gitignore` 强制提交。
