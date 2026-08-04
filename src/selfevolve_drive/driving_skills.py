from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .schema import Scenario


SKILL_NAMES = {
    "unsafe_following": "动态安全车距保持",
    "pedestrian_yield_failure": "行人冲突提前礼让",
    "speeding": "限速约束速度整形",
    "red_light_risk": "停止线渐进制动",
    "uncomfortable_motion": "低冲击轨迹平滑",
    "no_material_failure": "安全轨迹持续监控",
}

SKILL_ACTIONS = {
    "unsafe_following": ["预测前车未来位置", "建立三秒安全裕量", "限制纵向目标速度"],
    "pedestrian_yield_failure": ["沿标注轨迹预测行人占用区", "提前降低目标速度", "保留停止让行分支"],
    "speeding": ["读取道路限速", "加入安全速度缓冲", "裁剪整段速度曲线"],
    "red_light_risk": ["计算停止线剩余距离", "生成渐进制动曲线", "在停止线前保持零速"],
    "uncomfortable_motion": ["定位加速度与曲率突变", "约束相邻轨迹点变化", "复核舒适度得分"],
    "no_material_failure": ["保持当前轨迹", "持续监控风险证据", "风险升高时重新触发 Critic"],
}


def _memory_support(path: Path, causes: list[str]) -> dict[str, Any]:
    counts: Counter[int] = Counter()
    samples: list[str] = []
    matched = 0
    if path.is_file():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                row_causes = row.get("reflection", {}).get("root_causes", [])
                if not set(causes).intersection(row_causes):
                    continue
                matched += 1
                counts[int(row.get("round", 0))] += 1
                sample_id = str(row.get("sample_id", ""))
                if sample_id and sample_id not in samples and len(samples) < 4:
                    samples.append(sample_id)
    return {
        "matched_records": matched,
        "round_support": [{"round": key, "records": counts[key]} for key in sorted(counts)],
        "example_samples": samples,
    }


def _triggers(s: Scenario, primary: str) -> list[dict[str, str]]:
    weather = {"clear": "晴朗", "rain": "雨天", "fog": "雾天", "night": "夜间"}.get(s.weather, s.weather)
    route = {"straight": "直行", "left": "左转", "right": "右转"}.get(s.route_command, s.route_command)
    common = [{"label": "天气/导航", "value": f"{weather} · {route}"}]
    facts = {
        "unsafe_following": [{"label": "前车状态", "value": f"距离 {s.lead_distance:.1f} m · {s.lead_speed:.1f} m/s"}],
        "pedestrian_yield_failure": [{"label": "行人冲突", "value": f"距离 {s.pedestrian_distance:.1f} m · 自车 {s.ego_speed:.1f} m/s"}],
        "speeding": [{"label": "速度约束", "value": f"自车 {s.ego_speed:.1f} · 限速 {s.speed_limit:.1f} m/s"}],
        "red_light_risk": [{"label": "信号约束", "value": f"{s.traffic_light} · 停止线 {s.stopline_distance:.1f} m"}],
        "uncomfortable_motion": [{"label": "道路几何", "value": f"曲率 {s.road_curvature:.4f}"}],
        "no_material_failure": [{"label": "当前风险", "value": "Critic 未发现硬约束失败"}],
    }
    return facts.get(primary, []) + common


def _actions(s: Scenario, primary: str, revised: dict, source_record: dict | None) -> list[str]:
    target = revised["trajectory"]["target_speed"]
    pedestrian = (source_record or {}).get("source", {}).get("participants", {}).get("pedestrian") or {}
    dynamic = {
        "unsafe_following": [
            f"按前车 {s.lead_speed:.1f} m/s 预测未来位置",
            f"从当前 {s.lead_distance:.1f} m 间距建立三秒安全裕量",
            f"把重规划目标速度限制为 {target:.1f} m/s",
        ],
        "pedestrian_yield_failure": [
            f"沿 {pedestrian.get('duration_s', 0):.1f} s / {pedestrian.get('displacement_m', 0):.1f} m 标注轨迹预测占用区",
            f"在距行人 {s.pedestrian_distance:.1f} m 时将目标速度降至 {target:.1f} m/s",
            "保留停止让行分支并在 Critic 复评后放行",
        ],
        "speeding": [
            f"读取 {s.speed_limit:.1f} m/s 道路限速",
            "加入安全速度缓冲并裁剪整段速度曲线",
            f"复核重规划目标速度 {target:.1f} m/s",
        ],
        "red_light_risk": [
            f"计算距停止线 {s.stopline_distance:.1f} m 的剩余制动空间",
            f"生成目标速度 {target:.1f} m/s 的渐进制动曲线",
            "在停止线前保持零速并等待信号更新",
        ],
        "uncomfortable_motion": [
            f"定位曲率 {s.road_curvature:.4f} 下的加速度突变",
            "约束相邻轨迹点的速度与曲率变化",
            "重新计算 Comfort 并拒绝退化轨迹",
        ],
        "no_material_failure": ["保持当前轨迹", "持续监控风险证据", "风险升高时重新触发 Critic"],
    }
    return dynamic.get(primary, ["保持轨迹并监控不确定性"])


def build_driving_skill(
    scenario: Scenario,
    baseline: dict,
    revised: dict,
    memory_path: Path,
    source_record: dict | None = None,
) -> dict[str, Any]:
    reflection = baseline["reflection"]
    causes = list(reflection.get("root_causes") or ["no_material_failure"])
    primary = causes[0]
    memory = _memory_support(memory_path, causes)
    base_critic = baseline["critic"]
    revised_critic = revised["critic"]
    score_delta = revised_critic["overall_score"] - base_critic["overall_score"]
    safety_delta = revised_critic["safety_score"] - base_critic["safety_score"]
    target_delta = revised["trajectory"]["target_speed"] - baseline["trajectory"]["target_speed"]
    actions = _actions(scenario, primary, revised, source_record)
    signature = "|".join(sorted(causes) + SKILL_ACTIONS.get(primary, []))
    skill_id = "SKILL-" + hashlib.sha1(signature.encode("utf-8")).hexdigest()[:8].upper()
    validated = score_delta > 0 and safety_delta >= 0
    support_factor = min(0.12, memory["matched_records"] / 5000)
    confidence = min(0.99, float(reflection.get("confidence", .5)) + support_factor + (0.08 if validated else 0))

    source = (source_record or {}).get("source", {})
    pedestrian = source.get("participants", {}).get("pedestrian") or {}
    evidence_source = {
        "dataset": source.get("dataset", "interactive scenario"),
        "sample_token": source.get("sample_token"),
        "annotation_token": pedestrian.get("annotation_token"),
        "camera_count": len(source.get("image_refs", {})),
        "camera": "CAM_FRONT" if source.get("image_refs", {}).get("CAM_FRONT") else None,
    }
    stages = [
        {"phase": "失败捕获", "artifact": f"{len(causes)} 个根因", "detail": "Critic 从当前规划轨迹提取可核验失败证据"},
        {"phase": "反思归因", "artifact": SKILL_NAMES.get(primary, primary), "detail": "把失败根因映射为可执行的纠正策略"},
        {"phase": "Skill 抽象", "artifact": f"{len(actions)} 步动作", "detail": "从单场景参数抽象触发条件、动作序列与安全护栏"},
        {"phase": "重规划验证", "artifact": f"综合分 {score_delta:+.1f}", "detail": "在同一场景重新规划并由 Critic 实时复评"},
        {"phase": "记忆固化", "artifact": f"{memory['matched_records']} 条支持", "detail": "关联历史失败记忆，形成可检索、可复用驾驶 Skill"},
    ]
    return {
        "skill_id": skill_id,
        "name": SKILL_NAMES.get(primary, primary),
        "status": "validated" if validated else "candidate",
        "confidence": round(confidence, 3),
        "root_causes": causes,
        "triggers": _triggers(scenario, primary),
        "actions": actions,
        "guardrails": ["安全分不得下降", "规则分不得下降", "重新规划后必须再次经过 Critic"],
        "validation": {
            "overall_delta": round(score_delta, 3),
            "safety_delta": round(safety_delta, 3),
            "target_speed_delta": round(target_delta, 3),
            "passed": validated,
        },
        "memory": memory,
        "evidence_source": evidence_source,
        "generation_stages": stages,
    }
