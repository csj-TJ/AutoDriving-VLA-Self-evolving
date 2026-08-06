from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from selfevolve_drive.pipeline import load_config, read_jsonl
from selfevolve_drive.evaluation import evaluate_policy
from selfevolve_drive.schema import Scenario, Trajectory
from selfevolve_drive.self_evolution import run_self_evolution
from selfevolve_drive.training import fit_dpo, fit_reflection_sft, fit_sft_records, record_target_speed


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
    val = [r for r in records if r["split"] == "val"]

    def validation_metrics(policy):
        scenes = [Scenario(**r["scenario"]) for r in val]
        targets = [record_target_speed(r) for r in val]
        experts = [Trajectory(**r["expert_trajectory"]) for r in val]
        tokens = [r.get("source", {}).get("sample_token", r["sample_id"]) for r in val]
        metrics = evaluate_policy(policy, scenes, cfg["critic_weights"], targets, experts, tokens)
        metrics["selection_objective"] = round(
            metrics["target_speed_mae"] + .03 * (100.0 - metrics["overall_score"]), 4
        )
        return metrics

    started = time.perf_counter()
    ridge_grid = sorted(set([.01, .1, 1.0, 10.0, float(cfg["ridge_lambda"])]))
    sft_candidates = [(ridge, fit_sft_records(train, ridge=ridge)) for ridge in ridge_grid]
    reflection_candidates = [(ridge, fit_reflection_sft(train, ridge=ridge)) for ridge in ridge_grid]
    sft_trials = [{"ridge": ridge, "metrics": validation_metrics(policy)} for ridge, policy in sft_candidates]
    reflection_trials = [{"ridge": ridge, "metrics": validation_metrics(policy)} for ridge, policy in reflection_candidates]
    sft_choice = min(sft_trials, key=lambda item: item["metrics"]["selection_objective"])
    reflection_choice = min(reflection_trials, key=lambda item: item["metrics"]["selection_objective"])
    sft = next(policy for ridge, policy in sft_candidates if ridge == sft_choice["ridge"])
    reflection = next(policy for ridge, policy in reflection_candidates if ridge == reflection_choice["ridge"])
    policies = [sft, reflection]
    if not args.skip_dpo:
        beta_grid = sorted(set([.03, .08, .15, float(cfg["dpo_beta"])]))
        dpo_candidates = [(beta, fit_dpo(train, reflection, beta)) for beta in beta_grid]
        dpo_trials = [{"beta": beta, "metrics": validation_metrics(policy)} for beta, policy in dpo_candidates]
        dpo_choice = min(dpo_trials, key=lambda item: item["metrics"]["selection_objective"])
        policies.append(next(policy for beta, policy in dpo_candidates if beta == dpo_choice["beta"]))
    else:
        dpo_trials, dpo_choice = [], None
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
    training_summary = {
        "experiment": "nuScenes-mini lightweight Reflection self-evolution",
        "seed": cfg["seed"], "platform": platform.platform(),
        "train_accepted": len(train), "validation_rollouts": len(val),
        "ridge_search": {"sft": sft_trials, "reflection_sft": reflection_trials},
        "selected": {
            "sft_ridge": sft_choice["ridge"],
            "reflection_sft_ridge": reflection_choice["ridge"],
            "dpo_beta": dpo_choice["beta"] if dpo_choice else None,
        },
        "dpo_search": dpo_trials,
        "evolution_rounds": history,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(training_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "train_samples": len(train), "models": [p.name for p in policies],
        "evolution_rounds": len(history), "reflection_memory": len(memory),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
