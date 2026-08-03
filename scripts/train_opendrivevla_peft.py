from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def environment_status() -> dict:
    # The loop below does not call DeepSpeed directly, but the official
    # OpenDriveVLA model/vision code imports it while loading the base model.
    modules = ("torch", "transformers", "peft", "deepspeed", "mmcv", "mmdet3d")
    status = {name: importlib.util.find_spec(name) is not None for name in modules}
    cuda = False
    if status["torch"]:
        import torch
        cuda = torch.cuda.is_available()
    return {"platform": platform.system(), "modules": status, "cuda": cuda}


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenDriveVLA LoRA/Reflection-SFT/DPO training entry")
    parser.add_argument("--stage", choices=("sft", "reflection_sft", "dpo"), required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "opendrivevla_peft.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-only", action="store_true", help="Validate assets/config without loading CUDA")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    data_path = ROOT / cfg["stages"][args.stage]
    model_path = ROOT / cfg["base_model"]
    code_path = ROOT / cfg["official_code"]
    output = args.output or ROOT / "models" / f"opendrivevla_{args.stage}_lora"
    status = environment_status()
    plan = {
        "stage": args.stage, "base_model": str(model_path), "official_code": str(code_path),
        "data": str(data_path), "output": str(output), "environment": status,
        "trainable": "LoRA attention projections; vision tower frozen",
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    missing = [str(path) for path in (model_path / "model.safetensors", code_path / "llava", data_path) if not path.exists()]
    if missing:
        raise SystemExit("Missing required assets: " + ", ".join(missing))
    if args.check_only:
        return
    if platform.system() != "Linux" or not status["cuda"] or not all(status["modules"].values()):
        raise SystemExit("Full OpenDriveVLA PEFT requires Linux/CUDA and torch/transformers/peft/deepspeed/mmcv/mmdet3d. Use --check-only on Windows.")

    # Import only after the environment gate; Windows Demo remains dependency-free.
    sys.path.insert(0, str(code_path))
    import torch
    import torch.nn.functional as functional
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import DataLoader, Dataset
    from llava.model.builder import load_pretrained_model

    tokenizer, model, _, _ = load_pretrained_model(
        str(model_path), model_base=None, model_name="llava_qwen", device_map="cuda",
        multimodal=True, attn_implementation="sdpa",
        overwrite_config={"image_aspect_ratio": "pad", "vision_tower_test_mode": True},
    )
    for parameter in model.parameters():
        parameter.requires_grad = False
    peft_cfg = LoraConfig(
        r=cfg["lora_rank"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["target_modules"], task_type="CAUSAL_LM", bias="none",
    )
    model = get_peft_model(model, peft_cfg)
    model.train()

    rows = [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    class DrivingDataset(Dataset):
        def __len__(self): return len(rows)
        def __getitem__(self, index): return rows[index]

    def encode(prompt: str, response: str):
        prefix = tokenizer(prompt, add_special_tokens=True, truncation=True, max_length=cfg["max_length"])["input_ids"]
        full = tokenizer(prompt + "\n" + response, add_special_tokens=True, truncation=True, max_length=cfg["max_length"], return_tensors="pt")
        labels = full["input_ids"].clone(); labels[:, :min(len(prefix), labels.shape[1])] = -100
        return {"input_ids": full["input_ids"].cuda(), "attention_mask": full["attention_mask"].cuda(), "labels": labels.cuda()}

    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=cfg["learning_rate"])
    loader = DataLoader(DrivingDataset(), batch_size=1, shuffle=True, collate_fn=lambda batch: batch)
    accumulation = cfg["gradient_accumulation_steps"]
    step = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(cfg["epochs"]):
        for batch in loader:
            item = batch[0]
            if args.stage == "dpo":
                chosen = encode(item["prompt"], item["chosen"])
                rejected = encode(item["prompt"], item["rejected"])
                chosen_loss = model(**chosen).loss
                rejected_loss = model(**rejected).loss
                with torch.no_grad(), model.disable_adapter():
                    ref_chosen_loss = model(**chosen).loss
                    ref_rejected_loss = model(**rejected).loss
                policy_log_ratio = -chosen_loss + rejected_loss
                reference_log_ratio = -ref_chosen_loss + ref_rejected_loss
                loss = -functional.logsigmoid(cfg["dpo_beta"] * (policy_log_ratio - reference_log_ratio)).mean()
            else:
                encoded = encode(item["prompt"], item["response"])
                loss = model(**encoded).loss
            (loss / accumulation).backward()
            step += 1
            if step % accumulation == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
        print(json.dumps({"epoch": epoch + 1, "steps": step, "loss": float(loss.detach().cpu())}))
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    (output / "training_manifest.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
