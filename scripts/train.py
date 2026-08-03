from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from selfevolve_drive.pipeline import load_config, read_jsonl
from selfevolve_drive.schema import Scenario
from selfevolve_drive.self_evolution import run_self_evolution
from selfevolve_drive.training import fit_dpo, fit_reflection_sft, fit_sft_records


def main() -> None:
    p = argparse.ArgumentParser(description="Train SFT, Reflection-SFT and optional DPO lightweight policies")
    p.add_argument("--config", default=str(ROOT / "configs" / "default.json"))
    p.add_argument("--data", default=str(ROOT / "data" / "reflection_dataset.jsonl"))
    p.add_argument("--output-dir", default=str(ROOT / "models"))
    p.add_argument("--skip-dpo", action="store_true")
    p.add_argument("--evolution-rounds", type=int, default=3)
    args = p.parse_args()
    cfg, records = load_config(args.config), read_jsonl(args.data)
    train = [r for r in records if r["split"] == "train" and r["accepted"]]
    sft = fit_sft_records(train, ridge=cfg["ridge_lambda"])
    reflection = fit_reflection_sft(train, ridge=cfg["ridge_lambda"])
    policies = [sft, reflection]
    if not args.skip_dpo:
        policies.append(fit_dpo(train, reflection, cfg["dpo_beta"]))
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    for policy in policies:
        (out / f"{policy.name}.json").write_text(json.dumps(policy.to_dict(), indent=2), encoding="utf-8")
    evolution, history, memory = run_self_evolution(
        records, cfg["critic_weights"], cfg["ridge_lambda"], args.evolution_rounds,
    )
    for policy in evolution:
        (out / f"{policy.name}.json").write_text(json.dumps(policy.to_dict(), indent=2), encoding="utf-8")
    output_dir = ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evolution_history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_dir / "reflection_memory.jsonl").open("w", encoding="utf-8") as handle:
        for item in memory:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(json.dumps({
        "train_samples": len(train), "models": [p.name for p in policies],
        "evolution_rounds": len(history), "reflection_memory": len(memory),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
