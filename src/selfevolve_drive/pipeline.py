from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .critics import LLMCritic, RewardModelCritic, RuleBasedCritic
from .planner import Policy
from .reflection import reflect
from .schema import Scenario
from .simulator import generate_scenarios


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_records(config: dict[str, Any]) -> tuple[list[dict], RewardModelCritic]:
    scenes = generate_scenarios(config["num_samples"], config["seed"])
    planner = Policy("baseline", seed=config["seed"])
    rule = RuleBasedCritic(config["critic_weights"])
    raw: list[tuple[Scenario, Any, Any]] = []
    for s in scenes:
        t = planner.plan(s, config["trajectory_horizon"], config["dt"])
        raw.append((s, t, rule.evaluate(s, t)))
    reward = RewardModelCritic.fit(raw[: max(200, int(len(raw) * 0.5))], config["ridge_lambda"])
    llm = LLMCritic(rule, config["llm"])
    records: list[dict] = []
    for idx, (s, t, c) in enumerate(raw):
        chosen = c
        if idx % 10 == 1:
            chosen = llm.evaluate(s, t)
        elif idx % 10 == 2:
            chosen = reward.evaluate(s, t)
            # Preserve rule-localized failures for useful structured reflection.
            chosen.failures, chosen.evidence = c.failures, c.evidence + chosen.evidence
        refl = reflect(s, t, chosen)
        quality = float(round(0.65 * refl.confidence + 0.35 * min(1.0, len(chosen.evidence) / 2.0), 3))
        records.append({
            "sample_id": f"reflection-{idx:06d}", "scenario": s.to_dict(), "trajectory": t.to_dict(),
            "critic": chosen.to_dict(), "reflection": refl.to_dict(), "quality": quality,
            "accepted": bool(quality >= config["quality_threshold"]), "split": "test" if s.unseen else ("train" if idx % 10 < 8 else "val"),
        })
    return records, reward


def write_jsonl(path: str | Path, records: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
