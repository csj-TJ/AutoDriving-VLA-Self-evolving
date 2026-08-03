from __future__ import annotations

import importlib.util
import json
import math
import os
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .planner import Policy
from .schema import Scenario, Trajectory


MODEL_FILES = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "merges.txt",
    "vocab.json",
)
CODE_COMPONENTS = ("drivevla", "llava", "projects", "scripts", "third_party")


class BaseModelAdapter(Protocol):
    """Stable boundary for swapping the lightweight demo with a real VLA."""

    def plan(self, scenario: Scenario) -> Trajectory: ...

    def metadata(self) -> dict: ...


def _first_existing(paths: list[Path]) -> Path:
    return next((path for path in paths if path.exists()), paths[0])


def locate_opendrivevla(root: Path) -> tuple[Path, Path]:
    model_dir = root / "references" / "models" / "OpenDriveVLA-0.5B"
    code_dir = _first_existing([
        root / "references" / "repositories" / "OpenDriveVLA",
        root / "references" / "repos" / "OpenDriveVLA",
        root / "references" / "models" / "VLA-code" / "OpenDriveVLA-code",
    ])
    return model_dir, code_dir


def _safetensors_header_ok(path: Path) -> bool:
    """Validate the safetensors header without importing torch/safetensors."""
    try:
        with path.open("rb") as handle:
            header_len_raw = handle.read(8)
            if len(header_len_raw) != 8:
                return False
            header_len = struct.unpack("<Q", header_len_raw)[0]
            if not 2 <= header_len <= 100_000_000:
                return False
            header = json.loads(handle.read(header_len))
        return isinstance(header, dict) and any(k != "__metadata__" for k in header)
    except (OSError, ValueError, json.JSONDecodeError, struct.error):
        return False


def audit_opendrivevla(root: Path) -> dict:
    model_dir, code_dir = locate_opendrivevla(root)
    missing_model = [name for name in MODEL_FILES if not (model_dir / name).is_file()]
    missing_code = [name for name in CODE_COMPONENTS if not (code_dir / name).exists()]
    weight_path = model_dir / "model.safetensors"
    config_ok = False
    architecture = None
    try:
        config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        architecture = (config.get("architectures") or [None])[0]
        config_ok = architecture == "LlavaQwenForCausalLM"
    except (OSError, ValueError, json.JSONDecodeError):
        pass

    modules = ("torch", "transformers", "deepspeed", "mmcv", "mmdet", "mmdet3d")
    dependency_status = {name: importlib.util.find_spec(name) is not None for name in modules}
    cache_path = Path(os.getenv(
        "OPENDRIVEVLA_CACHE",
        str(root / "outputs" / "opendrivevla" / "demo_cache.jsonl"),
    ))
    model_complete = not missing_model and config_ok and _safetensors_header_ok(weight_path)
    code_complete = not missing_code
    runtime_ready = model_complete and code_complete and all(dependency_status.values())
    return {
        "model_dir": str(model_dir),
        "code_dir": str(code_dir),
        "model_complete": model_complete,
        "code_complete": code_complete,
        "checkpoint_installed": model_complete,
        "checkpoint_size_bytes": weight_path.stat().st_size if weight_path.exists() else 0,
        "safetensors_header_ok": _safetensors_header_ok(weight_path),
        "architecture": architecture,
        "missing_model_files": missing_model,
        "missing_code_components": missing_code,
        "dependencies": dependency_status,
        "runtime_ready": runtime_ready,
        "cache_path": str(cache_path),
        "cache_available": cache_path.is_file() and cache_path.stat().st_size > 0,
        "training_scripts_released": False,
    }


def parse_opendrivevla_trajectory(text: str, policy: str = "opendrivevla") -> Trajectory:
    """Convert the official six-point textual trajectory into the project schema."""
    match = re.search(r"<traj_start>(.*?)<traj_end>", text, flags=re.I | re.S)
    payload = match.group(1) if match else text
    pairs = re.findall(
        r"[\[(]\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*,\s*"
        r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*[\])]",
        payload,
    )
    if not pairs:
        raise ValueError("OpenDriveVLA output does not contain trajectory coordinates")
    xy = [(float(x), float(y)) for x, y in pairs[:6]]
    while len(xy) < 6:
        xy.append(xy[-1])
    points: list[list[float]] = []
    previous = (0.0, 0.0)
    speeds: list[float] = []
    for x, y in xy:
        speed = math.dist(previous, (x, y)) / 0.5
        speeds.append(speed)
        points.append([x, y, round(speed, 4)])
        previous = (x, y)
    return Trajectory(
        points=points,
        target_speed=round(sum(speeds[-3:]) / min(3, len(speeds)), 4),
        rationale="由 OpenDriveVLA 官方 <traj_start> 输出解析的 3 秒轨迹",
        policy=policy,
    )


@dataclass
class LiteVLAAdapter:
    """CPU surrogate that preserves the trajectory interface of a VLA planner."""

    root: Path
    policy_name: str = "reflection_sft"

    def _policy(self) -> Policy:
        if self.policy_name == "baseline":
            return Policy("baseline")
        path = self.root / "models" / f"{self.policy_name}.json"
        if not path.exists():
            return Policy("baseline")
        obj = json.loads(path.read_text(encoding="utf-8"))
        return Policy(obj["name"], obj["weights"], obj.get("seed", 42))

    def plan(self, scenario: Scenario) -> Trajectory:
        return self._policy().plan(scenario)

    def metadata(self) -> dict:
        audit = audit_opendrivevla(self.root)
        return {
            "runtime": "LiteVLA CPU surrogate",
            "runtime_mode": "lite",
            "policy": self.policy_name,
            "recommended_base_model": "OpenDriveVLA-0.5B",
            "required_for_full_experiment": True,
            "base_repository": "https://github.com/DriveVLA/OpenDriveVLA",
            "model_url": "https://huggingface.co/OpenDriveVLA/OpenDriveVLA-0.5B",
            "checkpoint_installed": audit["checkpoint_installed"],
            "weights_loaded": False,
            "runtime_ready": audit["runtime_ready"],
            "code_complete": audit["code_complete"],
            "cache_available": audit["cache_available"],
            "training_from_scratch": False,
            "status": "checkpoint installed; CPU mechanism fallback active" if audit["checkpoint_installed"] else "checkpoint missing; CPU mechanism fallback active",
            "disclosure": (
                "OpenDriveVLA checkpoint 已在本地核验，但当前请求由 CPU 轻量策略生成；"
                "这不是 OpenDriveVLA 实时视觉推理结果。"
            ),
        }


class CachedOpenDriveVLAAdapter:
    """Replay verified OpenDriveVLA GPU outputs while keeping the demo lightweight."""

    def __init__(self, root: Path, policy_name: str, fallback: BaseModelAdapter, strict: bool = False):
        self.root = root
        self.policy_name = policy_name
        self.fallback = fallback
        self.strict = strict
        self.audit = audit_opendrivevla(root)
        self.records = self._load(Path(self.audit["cache_path"]))
        self.last_source = "lite"

    @staticmethod
    def _load(path: Path) -> dict[str, list[dict]]:
        records: dict[str, list[dict]] = {}
        if not path.is_file():
            return records
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                    key = str(record.get("scene_id") or record.get("id"))
                    if key and key != "None":
                        records.setdefault(key, []).append(record)
                except (ValueError, json.JSONDecodeError):
                    continue
        return records

    def plan(self, scenario: Scenario) -> Trajectory:
        candidates = self.records.get(scenario.scene_id, [])
        aliases = {self.policy_name}
        if self.policy_name == "baseline":
            aliases.update({"opendrivevla", "opendrivevla_baseline"})
        record = next((item for item in candidates if str(
            item.get("policy") or (item.get("trajectory") or {}).get("policy") or
            ("opendrivevla_baseline" if item.get("answer") else "")
        ) in aliases), None)
        if record:
            self.last_source = "opendrivevla_cache"
            if isinstance(record.get("trajectory"), dict):
                obj = record["trajectory"]
                return Trajectory(
                    points=obj["points"], target_speed=float(obj["target_speed"]),
                    rationale=obj.get("rationale", "OpenDriveVLA cached trajectory"),
                    policy=obj.get("policy", self.policy_name),
                )
            answer = record.get("answer", "")
            if isinstance(answer, list):
                answer = answer[0] if answer else ""
            return parse_opendrivevla_trajectory(str(answer), self.policy_name)
        if self.strict:
            raise RuntimeError(
                f"No OpenDriveVLA cached trajectory for scene={scenario.scene_id}, policy={self.policy_name}. "
                "Run official GPU inference and import plan_conv.json first."
            )
        self.last_source = "lite"
        return self.fallback.plan(scenario)

    def metadata(self) -> dict:
        using_cache = self.last_source == "opendrivevla_cache"
        return {
            **self.fallback.metadata(),
            "runtime": "OpenDriveVLA GPU output cache" if using_cache else "Auto: LiteVLA fallback",
            "runtime_mode": self.last_source,
            "weights_loaded": False,
            "cache_available": bool(self.records),
            "cache_records": sum(len(items) for items in self.records.values()),
            "status": "verified OpenDriveVLA cached output" if using_cache else "no matching cached scene; CPU fallback active",
            "disclosure": (
                "轨迹来自 OpenDriveVLA 官方推理输出缓存；当前网页未实时加载 GPU 权重。"
                if using_cache else
                "本场景没有匹配的 OpenDriveVLA 推理缓存，已明确回退到 CPU 轻量策略。"
            ),
        }


def create_base_model(root: Path, policy_name: str = "reflection_sft", runtime: str = "auto") -> BaseModelAdapter:
    lite = LiteVLAAdapter(root=root, policy_name=policy_name)
    if runtime == "lite":
        return lite
    if runtime not in ("auto", "cache"):
        raise ValueError(f"unsupported runtime: {runtime}")
    return CachedOpenDriveVLAAdapter(root, policy_name, lite, strict=runtime == "cache")
