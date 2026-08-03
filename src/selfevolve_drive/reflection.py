from __future__ import annotations

from .schema import CriticResult, Reflection, Scenario, Trajectory


REMEDIES = {
    "unsafe_following": "reduce target speed and preserve at least a 3-second following margin",
    "pedestrian_yield_failure": "yield early and create a full-stop contingency before the crossing",
    "speeding": "cap the speed profile below the posted limit with a safety buffer",
    "red_light_risk": "start progressive braking before the stop line and hold at zero speed",
    "uncomfortable_motion": "smooth acceleration and curvature changes to reduce jerk",
}


def reflect(s: Scenario, t: Trajectory, c: CriticResult) -> Reflection:
    causes = c.failures or ["no_material_failure"]
    strategies = [REMEDIES.get(x, "retain the plan and monitor uncertainty") for x in causes]
    verdict = "revise" if c.overall_score < 80.0 or c.safety_score < 85.0 or c.rule_score < 85.0 else "accept"
    action = "; ".join(strategies) if verdict == "revise" else "keep the trajectory; no corrective update required"
    confidence = min(0.99, 0.55 + abs(float(c.overall_score) - 80.0) / 100.0 + 0.05 * len(c.evidence))
    return Reflection(verdict, causes, c.evidence, strategies, action, float(round(confidence, 3)))
