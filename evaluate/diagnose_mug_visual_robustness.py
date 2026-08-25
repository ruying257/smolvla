"""通过视觉扰动矩阵评测SmolVLA杯子策略的鲁棒性。

本工具面向 ``mug`` 杯子任务，在既有闭环评测（``evaluate.rollout_robustness``
薄副本的 ``run_single_rollout``）之上注入两类视觉扰动：

1. **外观变体**（环境级）：复用 ``MugTabletopEnv(appearance_variant=...)``
   替换杯子纹理（``original``/``green_white``/``changed``），物理完全一致。
2. **像素扰动**（图像级）：在相机图像捕获后、构造策略观测前，通过
   ``image_transform`` 注入亮度/对比度/Gamma/高斯噪声/高斯模糊/JPEG 压缩
   六种后处理，每种若干强度档。

协议：固定 ``scene_seed × task × prompt(canonical) × policy_seed``，只变化
扰动维度与强度，保证成功率差异可归因于扰动。输出逐条件 JSONL（断点续跑）、
按维度聚合的成功率与 Bootstrap 置信区间、崩溃阈值、鲁棒性曲线和 Markdown
报告。同一 ``--config`` 可分别用于 baseline 与域随机化训练 checkpoint，
产出对比（"评测发现 → 增强解决 → 复测验证"闭环）。

本脚本不修改 ``evaluate/rollout.py``，只依赖其薄副本 ``evaluate.rollout_robustness``。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from evaluate.common import (
    PROJECT_ROOT,
    find_pretrained_model,
    load_yaml_config,
    resolve_path,
    write_json,
    write_pillow_line_plot,
)
from evaluate.rollout_robustness import (  # 从薄副本导入，不经原 rollout.py
    RolloutResult,
    RolloutSpec,
    bootstrap_success_ci,
    load_policy_bundle,
    resolve_stage_detection,
    rollout_artifact_stem,
    run_single_rollout,
    sha256_file,
)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


CAMERAS = ("agent", "wrist")
PIXEL_PERTURBATIONS = (
    "brightness",
    "contrast",
    "gamma",
    "gaussian_noise",
    "gaussian_blur",
    "jpeg",
)
PERTURBATION_LABELS = {
    "brightness": "Brightness scale",
    "contrast": "Contrast factor",
    "gamma": "Gamma exponent",
    "gaussian_noise": "Gaussian noise sigma",
    "gaussian_blur": "Gaussian blur sigma",
    "jpeg": "JPEG quality",
}
# 环境级光照扰动：固定预设（default 基准 + alt 训练预设 + holdout 仅评测预设）。
# 评测矩阵 = appearance_variants × lighting_presets 全交叉；holdout_presets 只用于
# 评测（不参与训练），验证未见光照泛化。
LIGHTING_PRESET_KEYS = {"a_scale", "b_azimuth_deg", "c_scale"}
RESULTS_JSONL = "rollouts.jsonl"
MANIFEST_JSON = "run_manifest.json"
AGGREGATE_CSV = "robustness_aggregate.csv"
DETAIL_CSV = "rollout_detail.csv"
SUMMARY_JSON = "summary.json"
REPORT_MD = "report.md"
CURVE_DIR = "curves"
EXECUTION_HORIZON = 50
IMAGE_SHAPE = (256, 256, 3)


@dataclass(frozen=True)
class PerturbationSpec:
    """一个像素扰动条件：名称与强度档。"""

    name: str
    intensity: float

    @property
    def label(self) -> str:
        return f"{self.name}-{self.intensity}"


@dataclass(frozen=True)
class RobustnessCondition:
    """一条鲁棒性评测条件：固定 scene/task/policy，只变视觉条件。"""

    scene_seed: int
    task_id: str
    policy_seed: int
    appearance_variant: str = "original"
    lighting_preset: str = "default"
    lighting_params: dict[str, float] | None = None
    perturbation: PerturbationSpec | None = None

    @property
    def key(self) -> str:
        if self.perturbation is not None:
            return (
                f"scene={self.scene_seed}|task={self.task_id}|"
                f"pert={self.perturbation.label}|policy={self.policy_seed}"
            )
        return (
            f"scene={self.scene_seed}|task={self.task_id}|"
            f"app={self.appearance_variant}|light={self.lighting_preset}|"
            f"policy={self.policy_seed}"
        )

    @property
    def is_holdout(self) -> bool:
        """该条件是否使用 holdout 光照预设（由外层预设集合判定，默认 False）。"""
        return False


def format_intensity_component(intensity: float) -> str:
    """把扰动强度格式化为稳定、简洁的目录名。"""
    return format(float(intensity), ".15g")


def condition_artifact_subdir(condition: RobustnessCondition) -> Path:
    """返回视觉条件对应的视频和动作日志相对子目录。"""
    if condition.perturbation is not None:
        return (
            Path("pixel")
            / condition.perturbation.name
            / format_intensity_component(condition.perturbation.intensity)
        )
    return Path("appearance") / condition.appearance_variant / condition.lighting_preset


# ---------------------------------------------------------------------------
# 像素扰动实现（输入输出均为 uint8 (256,256,3)）
# ---------------------------------------------------------------------------


def apply_brightness(image: NDArray[np.uint8], scale: float) -> NDArray[np.uint8]:
    """亮度乘子扰动。"""
    return np.clip(image.astype(np.int16) * scale, 0, 255).astype(np.uint8)


def apply_contrast(image: NDArray[np.uint8], factor: float) -> NDArray[np.uint8]:
    """以 128 为中心的对比度扰动。"""
    return np.clip((image.astype(np.int16) - 128) * factor + 128, 0, 255).astype(np.uint8)


def apply_gamma(image: NDArray[np.uint8], gamma: float) -> NDArray[np.uint8]:
    """Gamma 指数扰动。"""
    normalized = image.astype(np.float64) / 255.0
    return np.clip(255.0 * np.power(normalized, gamma), 0, 255).astype(np.uint8)


def apply_gaussian_noise(
    image: NDArray[np.uint8],
    sigma: float,
    rng: np.random.Generator,
) -> NDArray[np.uint8]:
    """高斯加性噪声（uint8 空间，逐帧独立采样）。"""
    noise = rng.normal(0.0, float(sigma), size=image.shape)
    return np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def apply_gaussian_blur(image: NDArray[np.uint8], sigma: float) -> NDArray[np.uint8]:
    """高斯模糊（Pillow）。"""
    from PIL import Image, ImageFilter

    blurred = Image.fromarray(image).filter(ImageFilter.GaussianBlur(radius=float(sigma)))
    return np.asarray(blurred, dtype=np.uint8).copy()


def apply_jpeg(image: NDArray[np.uint8], quality: int) -> NDArray[np.uint8]:
    """JPEG 重压缩伪影（Pillow 编码再解码）。"""
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="JPEG", quality=int(quality))
    buffer.seek(0)
    return np.asarray(Image.open(buffer).convert("RGB"), dtype=np.uint8).copy()


def apply_pixel_perturbation(
    image: NDArray[np.uint8],
    name: str,
    intensity: float,
    rng: np.random.Generator,
) -> NDArray[np.uint8]:
    """按名称分发像素扰动并统一校验输入输出格式。"""
    array = np.asarray(image)
    if array.shape != IMAGE_SHAPE or array.dtype != np.uint8:
        raise ValueError(f"像素扰动输入必须是{IMAGE_SHAPE} uint8，实际={array.shape}/{array.dtype}")
    if name == "brightness":
        output = apply_brightness(array, intensity)
    elif name == "contrast":
        output = apply_contrast(array, intensity)
    elif name == "gamma":
        output = apply_gamma(array, intensity)
    elif name == "gaussian_noise":
        output = apply_gaussian_noise(array, intensity, rng)
    elif name == "gaussian_blur":
        output = apply_gaussian_blur(array, intensity)
    elif name == "jpeg":
        output = apply_jpeg(array, intensity)
    else:
        raise ValueError(f"未知像素扰动: {name!r}，可选值={PIXEL_PERTURBATIONS}")
    if output.shape != IMAGE_SHAPE or output.dtype != np.uint8:
        raise ValueError(f"像素扰动输出必须是{IMAGE_SHAPE} uint8，实际={output.shape}/{output.dtype}")
    return output


def stable_condition_seed(condition: RobustnessCondition) -> int:
    """从条件派生确定性随机种子（不依赖全局 np.random 状态）。"""
    assert condition.perturbation is not None
    payload = (
        f"scene={condition.scene_seed}|task={condition.task_id}|"
        f"policy={condition.policy_seed}|pert={condition.perturbation.label}"
    )
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def build_image_transform(
    condition: RobustnessCondition,
) -> Callable[[dict[str, NDArray[np.uint8]]], dict[str, NDArray[np.uint8]]]:
    """构造注入 run_single_rollout 的 image_transform。

    随机扰动（gaussian_noise）使用条件派生的确定性 Generator，随 rollout
    步进顺序消耗；同一条件下 rollout 步数确定（策略与场景可复现），因此
    噪声序列可复现。非随机扰动不消耗随机数。
    """
    assert condition.perturbation is not None
    name = condition.perturbation.name
    intensity = condition.perturbation.intensity
    rng = np.random.default_rng(stable_condition_seed(condition))

    def transform(images: dict[str, NDArray[np.uint8]]) -> dict[str, NDArray[np.uint8]]:
        if set(images) != set(CAMERAS):
            raise ValueError(f"image_transform输入相机键必须为{CAMERAS}，实际为{set(images)}")
        return {
            key: apply_pixel_perturbation(image, name, intensity, rng)
            for key, image in images.items()
        }

    return transform


# ---------------------------------------------------------------------------
# 配置解析与条件构建
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """创建视觉鲁棒性评测命令行解析器。"""
    parser = argparse.ArgumentParser(description="评测SmolVLA杯子策略对视觉扰动的鲁棒性")
    parser.add_argument("--checkpoint", type=Path, required=True, help="模型、checkpoint或训练输出目录")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "diagnose_mug_robustness.yaml",
        help="鲁棒性评测YAML配置",
    )
    parser.add_argument("--output-dir", type=Path, help="覆盖配置中的评测输出目录")
    parser.add_argument("--device", default="cuda", help="推理设备，默认cuda")
    parser.add_argument("--max-scenes", type=int, help="仅运行前N个scene，用于冒烟")
    parser.add_argument(
        "--perturbations",
        help="逗号分隔的像素扰动维度白名单，例如 brightness,gamma",
    )
    parser.add_argument("--skip-appearance", action="store_true", help="跳过外观变体条件")
    return parser


def parse_config(config: dict[str, Any]) -> dict[str, Any]:
    """读取并校验鲁棒性评测配置。"""
    section = config.get("diagnostic")
    if not isinstance(section, dict):
        raise ValueError("配置必须包含diagnostic映射")
    environment = str(section.get("environment", "mug"))
    if environment != "mug":
        raise ValueError(f"本工具只支持mug环境，实际为{environment!r}")
    scene_seeds = [int(value) for value in section.get("scene_seeds", [])]
    task_ids = [str(value) for value in section.get("task_ids", [])]
    prompt_type = str(section.get("prompt_type", "canonical"))
    policy_seeds = [int(value) for value in section.get("policy_seeds", [])]
    appearance_variants = [str(value) for value in section.get("appearance_variants", [])]
    raw_pixel = section.get("pixel_perturbations", {})
    raw_lighting = section.get("lighting_presets", {})
    raw_holdout = section.get("holdout_presets", [])
    fps = int(section.get("fps", 20))
    max_steps = int(section.get("max_steps", 360))
    drop_pp = float(section.get("collapse_baseline_drop_pp", 20))

    if not scene_seeds or len(scene_seeds) != len(set(scene_seeds)):
        raise ValueError("scene_seeds必须非空且不重复")
    if not task_ids or any(task not in {"mug_on_blue", "mug_on_yellow"} for task in task_ids):
        raise ValueError("task_ids必须锁定为mug_on_blue/mug_on_yellow")
    if prompt_type != "canonical":
        raise ValueError("杯子鲁棒性评测只支持canonical措辞")
    if policy_seeds != [20260]:
        raise ValueError("本工具锁定policy_seed=20260以隔离扰动变量")
    pixel_perturbations: dict[str, list[float]] = {}
    if raw_pixel:
        if not isinstance(raw_pixel, dict):
            raise ValueError("pixel_perturbations必须是映射")
        unknown = set(raw_pixel) - set(PIXEL_PERTURBATIONS)
        if unknown:
            raise ValueError(f"未知像素扰动维度: {sorted(unknown)}")
        for name, intensities in raw_pixel.items():
            values = [float(value) for value in intensities]
            if not values or len(values) != len(set(values)):
                raise ValueError(f"扰动{name}的强度必须非空且不重复")
            if not np.isfinite(values).all():
                raise ValueError(f"扰动{name}的强度必须为有限数")
            pixel_perturbations[name] = values
    lighting_presets: dict[str, dict[str, float]] = {}
    if raw_lighting:
        if not isinstance(raw_lighting, dict):
            raise ValueError("lighting_presets必须是映射")
        for name, params in raw_lighting.items():
            if not isinstance(params, dict) or set(params) != LIGHTING_PRESET_KEYS:
                raise ValueError(f"光照预设{name!r}必须含a_scale/b_azimuth_deg/c_scale")
            values = {key: float(value) for key, value in params.items()}
            if not all(np.isfinite(value) for value in values.values()):
                raise ValueError(f"光照预设{name!r}参数必须为有限数")
            lighting_presets[name] = values
    holdout_presets = [str(name) for name in raw_holdout]
    unknown_holdout = set(holdout_presets) - set(lighting_presets)
    if unknown_holdout:
        raise ValueError(f"holdout_presets引用未知预设: {sorted(unknown_holdout)}")
    if fps <= 0 or max_steps <= 0:
        raise ValueError("fps和max_steps必须大于零")
    if not np.isfinite(drop_pp) or not 0.0 <= drop_pp <= 100.0:
        raise ValueError("collapse_baseline_drop_pp必须位于[0,100]")
    return {
        "environment": environment,
        "output_dir": str(section.get("output_dir", "outputs/eval/mug_robustness")),
        "fps": fps,
        "max_steps": max_steps,
        "scene_seeds": scene_seeds,
        "task_ids": task_ids,
        "prompt_type": prompt_type,
        "policy_seeds": policy_seeds,
        "appearance_variants": appearance_variants,
        "pixel_perturbations": pixel_perturbations,
        "lighting_presets": lighting_presets,
        "holdout_presets": holdout_presets,
        "collapse_baseline_drop_pp": drop_pp,
    }


def build_conditions(settings: dict[str, Any]) -> list[RobustnessCondition]:
    """按外观变体×光照预设全交叉 + 像素扰动枚举完整评测矩阵。"""
    conditions: list[RobustnessCondition] = []
    lighting_presets = settings.get("lighting_presets", {})
    presets = list(lighting_presets) or ["default"]
    for scene_seed in settings["scene_seeds"]:
        for task_id in settings["task_ids"]:
            for policy_seed in settings["policy_seeds"]:
                for variant in settings["appearance_variants"]:
                    for preset in presets:
                        conditions.append(
                            RobustnessCondition(
                                scene_seed=scene_seed,
                                task_id=task_id,
                                policy_seed=policy_seed,
                                appearance_variant=variant,
                                lighting_preset=preset,
                                lighting_params=dict(lighting_presets[preset])
                                if preset in lighting_presets
                                else None,
                            )
                        )
                for name, intensities in settings["pixel_perturbations"].items():
                    for intensity in intensities:
                        conditions.append(
                            RobustnessCondition(
                                scene_seed=scene_seed,
                                task_id=task_id,
                                policy_seed=policy_seed,
                                perturbation=PerturbationSpec(name=name, intensity=intensity),
                            )
                        )
    keys = [condition.key for condition in conditions]
    if len(keys) != len(set(keys)):
        raise RuntimeError("条件矩阵存在重复键")
    return conditions


# ---------------------------------------------------------------------------
# 审计
# ---------------------------------------------------------------------------


def verify_pixel_perturbation(
    original: NDArray[np.uint8],
    perturbed: NDArray[np.uint8],
    name: str,
    intensity: float,
) -> dict[str, Any]:
    """校验像素扰动确实生效且保持格式。"""
    if original.shape != perturbed.shape or original.dtype != perturbed.dtype:
        raise RuntimeError(f"扰动{name}改变了图像格式")
    changed = int(np.count_nonzero(original != perturbed))
    if changed == 0:
        raise RuntimeError(f"扰动{name}-{intensity}未引起任何像素变化")
    total = int(original.size)
    return {
        "name": name,
        "intensity": float(intensity),
        "changed_pixels": changed,
        "changed_fraction": changed / total if total else 0.0,
    }


def verify_appearance_variant(
    project_root: Path,
    variants: Sequence[str],
) -> dict[str, Any]:
    """校验外观变体：纹理存在、SHA-256 互异。"""
    from sim.mug_environment import MUG_APPEARANCE_TEXTURES, resolve_mug_texture_path

    asset_root = project_root / "assets" / "mujoco"
    missing = [variant for variant in variants if variant not in MUG_APPEARANCE_TEXTURES]
    if missing:
        raise RuntimeError(f"外观变体未注册: {missing}")
    digests: dict[str, str] = {}
    for variant in variants:
        texture_path = resolve_mug_texture_path(asset_root, variant)
        digests[variant] = sha256_file(texture_path)
    unique = set(digests.values())
    if len(unique) != len(variants):
        raise RuntimeError(f"外观变体纹理SHA-256存在重复: {digests}")
    return {"texture_sha256": digests}


# ---------------------------------------------------------------------------
# 执行
# ---------------------------------------------------------------------------


def _load_completed_keys(output_dir: Path) -> set[str]:
    """读取已有JSONL中的条件键，用于断点续跑。"""
    path = output_dir / RESULTS_JSONL
    if not path.is_file():
        return set()
    keys: set[str] = set()
    with path.open("r", encoding="utf-8", newline="\n") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = record.get("rollout_key") or record.get("condition_key")
            if key:
                keys.add(str(key))
    return keys


def run_diagnostic(
    checkpoint: Path,
    config: dict[str, Any],
    settings: dict[str, Any],
    output_dir: Path,
    device: str,
    max_scenes: int | None,
    perturbation_filter: set[str] | None,
    skip_appearance: bool,
) -> dict[str, Any]:
    """执行完整鲁棒性扰动评测矩阵。"""
    import torch

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("请求cuda评测，但当前环境没有可用CUDA")

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = find_pretrained_model(checkpoint)
    checkpoint_sha256 = sha256_file(checkpoint / "model.safetensors")
    appearance_audit = (
        verify_appearance_variant(PROJECT_ROOT, settings["appearance_variants"])
        if not skip_appearance and settings["appearance_variants"]
        else {"texture_sha256": {}}
    )

    pixel_perturbations = settings["pixel_perturbations"]
    if perturbation_filter is not None:
        unknown = perturbation_filter - set(pixel_perturbations)
        if unknown:
            raise ValueError(f"扰动过滤包含未知维度: {sorted(unknown)}")
        pixel_perturbations = {
            name: intensities
            for name, intensities in pixel_perturbations.items()
            if name in perturbation_filter
        }
    if not pixel_perturbations and skip_appearance and not settings["lighting_presets"]:
        raise ValueError("同时跳过外观与像素扰动且无光照预设时没有可评测条件")

    settings_effective = dict(settings)
    settings_effective["pixel_perturbations"] = pixel_perturbations
    settings_effective["appearance_variants"] = (
        [] if skip_appearance else settings["appearance_variants"]
    )
    scene_seeds = settings["scene_seeds"][:max_scenes] if max_scenes is not None else settings["scene_seeds"]
    settings_effective["scene_seeds"] = scene_seeds

    conditions = build_conditions(settings_effective)
    if not conditions:
        raise ValueError("配置后没有可评测条件")

    # 像素扰动静态有效性自检：对固定参考图验证每档强度确实改变像素。
    pixel_audits: list[dict[str, Any]] = []
    if pixel_perturbations:
        reference = np.random.default_rng(20260813).integers(0, 256, size=IMAGE_SHAPE, dtype=np.uint8)
        for name, intensities in pixel_perturbations.items():
            for intensity in intensities:
                rng = np.random.default_rng(20260813)
                perturbed = apply_pixel_perturbation(reference, name, intensity, rng)
                pixel_audits.append(verify_pixel_perturbation(reference, perturbed, name, intensity))

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "device": device,
        "config": config,
        "conditions": [condition.key for condition in conditions],
        "condition_count": len(conditions),
        "scene_seeds": scene_seeds,
        "task_ids": settings["task_ids"],
        "policy_seeds": settings["policy_seeds"],
        "prompt_type": settings["prompt_type"],
        "appearance_variants": settings_effective["appearance_variants"],
        "appearance_texture_sha256": appearance_audit["texture_sha256"],
        "pixel_perturbations": {
            name: intensities for name, intensities in pixel_perturbations.items()
        },
        "pixel_perturbation_audits": pixel_audits,
        "lighting_presets": settings["lighting_presets"],
        "holdout_presets": settings["holdout_presets"],
        "fps": settings["fps"],
        "max_steps": settings["max_steps"],
        "execution_horizon": EXECUTION_HORIZON,
        "collapse_baseline_drop_pp": settings["collapse_baseline_drop_pp"],
    }
    write_json(output_dir / MANIFEST_JSON, manifest)

    completed = _load_completed_keys(output_dir)
    pending = [condition for condition in conditions if condition.key not in completed]
    print(
        f"条件总数={len(conditions)}，已完成={len(conditions) - len(pending)}，待运行={len(pending)}",
        flush=True,
    )

    stage_detection = resolve_stage_detection({})
    results: list[RolloutResult] = []
    if pending:
        policy, preprocessor, postprocessor = load_policy_bundle(
            checkpoint,
            device,
            EXECUTION_HORIZON,
        )
        with (output_dir / RESULTS_JSONL).open("a", encoding="utf-8", newline="\n") as jsonl:
            for condition in pending:
                spec = RolloutSpec(
                    scene_seed=condition.scene_seed,
                    task_id=condition.task_id,
                    prompt_type=settings["prompt_type"],
                    policy_seed=condition.policy_seed,
                )
                # 同一 scene/task/policy 下多个扰动条件共用默认 artifact_stem，
                # 必须用条件专属前缀覆盖，避免视频/动作日志互相覆盖。
                # 后缀只取扰动维度（含 | 的 condition.key 不能用于文件名）。
                if condition.perturbation is not None:
                    suffix = (
                        f"pert_{condition.perturbation.name}_"
                        f"{condition.perturbation.intensity}"
                    )
                else:
                    suffix = (
                        f"app_{condition.appearance_variant}_"
                        f"light_{condition.lighting_preset}"
                    )
                artifact_stem_override = f"{rollout_artifact_stem(spec)}__{suffix}"
                artifact_subdir_override = condition_artifact_subdir(condition)
                if condition.perturbation is not None:
                    result = run_single_rollout(
                        policy,
                        preprocessor,
                        postprocessor,
                        spec,
                        output_dir,
                        settings["fps"],
                        settings["max_steps"],
                        device,
                        checkpoint_sha256,
                        execution_horizon=EXECUTION_HORIZON,
                        environment="mug",
                        appearance_variant="original",
                        stage_detection_settings=stage_detection,
                        image_transform=build_image_transform(condition),
                        artifact_stem_override=artifact_stem_override,
                        artifact_subdir_override=artifact_subdir_override,
                    )
                else:
                    result = run_single_rollout(
                        policy,
                        preprocessor,
                        postprocessor,
                        spec,
                        output_dir,
                        settings["fps"],
                        settings["max_steps"],
                        device,
                        checkpoint_sha256,
                        execution_horizon=EXECUTION_HORIZON,
                        environment="mug",
                        appearance_variant=condition.appearance_variant,
                        lighting=condition.lighting_params,
                        stage_detection_settings=stage_detection,
                        artifact_stem_override=artifact_stem_override,
                        artifact_subdir_override=artifact_subdir_override,
                    )
                tagged = asdict(result)
                tagged["condition_key"] = condition.key
                tagged["appearance_variant"] = condition.appearance_variant
                tagged["lighting_preset"] = condition.lighting_preset
                tagged["holdout"] = condition.lighting_preset in settings["holdout_presets"]
                tagged["perturbation_name"] = (
                    condition.perturbation.name if condition.perturbation is not None else None
                )
                tagged["perturbation_intensity"] = (
                    float(condition.perturbation.intensity)
                    if condition.perturbation is not None
                    else None
                )
                jsonl.write(json.dumps(tagged, ensure_ascii=False) + "\n")
                jsonl.flush()
                print(json.dumps(tagged, ensure_ascii=False), flush=True)

    # 汇总：重新读取全部完成记录（含本次与历史断点）。
    results = load_results_jsonl(output_dir / RESULTS_JSONL)
    summary = build_summary(results, settings_effective)
    write_json(output_dir / SUMMARY_JSON, summary)
    write_csv(output_dir / DETAIL_CSV, summary["detail_rows"])
    write_csv(output_dir / AGGREGATE_CSV, summary["aggregate_rows"])
    write_report(output_dir / REPORT_MD, summary)
    write_curves(output_dir, summary)
    return summary


def load_results_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取全部完成记录并转回结构化字典。"""
    records: list[dict[str, Any]] = []
    if not path.is_file():
        return records
    with path.open("r", encoding="utf-8", newline="\n") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# 汇总与报告
# ---------------------------------------------------------------------------


def success_rate(results: list[dict[str, Any]]) -> float:
    """计算一组记录的成功率。"""
    return sum(1 for record in results if record.get("success")) / len(results) if results else 0.0


def collapse_threshold(
    intensities: Sequence[float],
    rates: Sequence[float],
    baseline_rate: float,
    drop_pp: float,
) -> float | str:
    """返回首次跌破阈值的最低强度档；未跌破返回 ``"not_collapsed"``。

    阈值定义为 ``max(0.5, baseline_rate - drop_pp/100)``。
    """
    threshold = max(0.5, baseline_rate - drop_pp / 100.0)
    for intensity, rate in zip(intensities, rates, strict=True):
        if rate < threshold:
            return float(intensity)
    return "not_collapsed"


def build_summary(
    records: list[dict[str, Any]],
    settings: dict[str, Any],
) -> dict[str, Any]:
    """聚合明细行、维度×强度行与报告摘要。"""
    detail_rows: list[dict[str, Any]] = []
    for record in records:
        detail_rows.append(
            {
                "condition_key": record.get("condition_key") or record.get("rollout_key"),
                "scene_seed": record.get("scene_seed"),
                "task_id": record.get("task_id"),
                "appearance_variant": record.get("appearance_variant"),
                "lighting_preset": record.get("lighting_preset"),
                "holdout": bool(record.get("holdout", False)),
                "perturbation_name": record.get("perturbation_name"),
                "perturbation_intensity": record.get("perturbation_intensity"),
                "success": bool(record.get("success")),
                "failure_mode": record.get("failure_mode"),
                "steps": record.get("steps"),
                "latency_mean_ms": record.get("latency_mean_ms"),
            }
        )

    # 基线：外观 original + 光照 default（无像素扰动）的全部记录。
    baseline_records = [
        record
        for record in records
        if record.get("appearance_variant") == "original"
        and record.get("lighting_preset") in (None, "default")
    ]
    baseline_rate = success_rate(baseline_records)
    baseline_ci = [0.0, 0.0]
    if baseline_records:
        baseline_ci = bootstrap_success_ci(
            [_record_to_result(record) for record in baseline_records]
        )

    aggregate_rows: list[dict[str, Any]] = []
    # 外观×光照组合行：每个 (变体, 预设) 一组；相对基线标记崩溃。
    combo_collapse: dict[str, str] = {}
    for variant in settings["appearance_variants"]:
        presets = list(settings["lighting_presets"]) or ["default"]
        for preset in presets:
            group = [
                record
                for record in records
                if record.get("appearance_variant") == variant
                and record.get("lighting_preset") in (None, preset)
            ]
            if not group:
                continue
            rate = success_rate(group)
            ci = bootstrap_success_ci([_record_to_result(record) for record in group])
            label = f"{variant}@{preset}"
            collapsed = ""
            if not (variant == "original" and preset == "default"):
                threshold = max(0.5, baseline_rate - settings["collapse_baseline_drop_pp"] / 100.0)
                if rate < threshold:
                    collapsed = "yes"
                    combo_collapse[label] = (
                        f"below max(0.5, baseline-{settings['collapse_baseline_drop_pp']:.0f}pp)"
                    )
            aggregate_rows.append(
                {
                    "perturbation": "appearance",
                    "intensity": label,
                    "successes": sum(1 for record in group if record.get("success")),
                    "rollouts": len(group),
                    "success_rate": rate,
                    "ci_low": ci[0],
                    "ci_high": ci[1],
                    "collapsed": "baseline" if label == "original@default" else collapsed,
                    "holdout": preset in settings["holdout_presets"],
                }
            )

    # 像素扰动行：每个维度按强度升序一组。
    pixel_collapse: dict[str, Any] = {}
    for name, intensities in settings["pixel_perturbations"].items():
        group_rates: list[float] = []
        for intensity in intensities:
            group = [
                record
                for record in records
                if record.get("perturbation_name") == name
                and record.get("perturbation_intensity") is not None
                and abs(float(record["perturbation_intensity"]) - float(intensity)) < 1e-9
            ]
            if not group:
                continue
            rate = success_rate(group)
            ci = bootstrap_success_ci([_record_to_result(record) for record in group])
            group_rates.append(rate)
            aggregate_rows.append(
                {
                    "perturbation": name,
                    "intensity": float(intensity),
                    "successes": sum(1 for record in group if record.get("success")),
                    "rollouts": len(group),
                    "success_rate": rate,
                    "ci_low": ci[0],
                    "ci_high": ci[1],
                    "collapsed": "",
                }
            )
        if group_rates:
            threshold_value = collapse_threshold(
                intensities,
                group_rates,
                baseline_rate,
                settings["collapse_baseline_drop_pp"],
            )
            pixel_collapse[name] = {
                "collapse_intensity": threshold_value,
                "threshold_rule": (
                    f"max(0.5, baseline-{settings['collapse_baseline_drop_pp']:.0f}pp)"
                ),
            }
            for row in aggregate_rows:
                if row["perturbation"] == name and row["intensity"] == threshold_value:
                    row["collapsed"] = "yes"

    # 阶段摘要：读取每条 action_trace 的最高直接阶段（可选，失败时忽略）。
    stage_summary: dict[str, dict[str, int]] = {}
    for record in records:
        if record.get("perturbation_name") is not None:
            name = record["perturbation_name"]
            label = str(record["perturbation_intensity"])
        else:
            name = record.get("appearance_variant", "original")
            label = str(record.get("lighting_preset", "default"))
        stage = read_highest_stage(record.get("action_trace_path"))
        bucket = stage_summary.setdefault(f"{name}-{label}", {})
        bucket[stage] = bucket.get(stage, 0) + 1

    return {
        "condition_count": len(records),
        "baseline_rate": baseline_rate,
        "baseline_ci_low": baseline_ci[0],
        "baseline_ci_high": baseline_ci[1],
        "pixel_collapse": pixel_collapse,
        "combo_collapse": combo_collapse,
        "detail_rows": detail_rows,
        "aggregate_rows": aggregate_rows,
        "stage_summary": stage_summary,
    }


def _record_to_result(record: dict[str, Any]) -> RolloutResult:
    """把JSONL记录转换为RolloutResult（Bootstrap复用）。"""
    return RolloutResult(
        rollout_key=str(record.get("rollout_key") or record.get("condition_key")),
        scene_seed=int(record["scene_seed"]),
        policy_seed=int(record["policy_seed"]),
        task_id=str(record["task_id"]),
        task=str(record.get("task", "")),
        prompt_type=str(record.get("prompt_type", "canonical")),
        success=bool(record.get("success")),
        failure_mode=str(record.get("failure_mode", "timeout")),
        steps=int(record.get("steps", 0)),
        elapsed_seconds=float(record.get("elapsed_seconds", 0.0)),
        latency_mean_ms=float(record.get("latency_mean_ms", 0.0)),
        latency_p95_ms=float(record.get("latency_p95_ms", 0.0)),
        clipped_action_steps=int(record.get("clipped_action_steps", 0)),
        clipped_action_rate=float(record.get("clipped_action_rate", 0.0)),
        action_trace_path=str(record.get("action_trace_path", "")),
        checkpoint_sha256=str(record.get("checkpoint_sha256", "")),
        video_path=str(record.get("video_path", "")),
        video_retained=bool(record.get("video_retained", False)),
        error=str(record.get("error", "")),
        completed_at=str(record.get("completed_at", "")),
    )


def read_highest_stage(action_trace_path: str | None) -> str:
    """读取action trace最后一帧的 ``highest_direct_stage``。"""
    if not action_trace_path:
        return "none"
    path = Path(action_trace_path)
    if not path.is_file():
        return "none"
    last_stage = "none"
    try:
        with path.open("r", encoding="utf-8", newline="\n") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                stage = record.get("highest_direct_stage")
                if stage:
                    last_stage = str(stage)
    except (OSError, json.JSONDecodeError):
        return "none"
    return last_stage


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """写入UTF-8 CSV。"""
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_curves(output_dir: Path, summary: dict[str, Any]) -> None:
    """为每个像素扰动维度绘制成功率-强度曲线。"""
    curve_dir = output_dir / CURVE_DIR
    curve_dir.mkdir(parents=True, exist_ok=True)
    by_perturbation: dict[str, list[dict[str, Any]]] = {}
    for row in summary["aggregate_rows"]:
        if row["perturbation"] == "appearance":
            continue
        by_perturbation.setdefault(row["perturbation"], []).append(row)
    for name, rows in by_perturbation.items():
        rows.sort(key=lambda row: float(row["intensity"]))
        values = np.asarray([float(row["success_rate"]) for row in rows], dtype=np.float64)
        write_pillow_line_plot(
            curve_dir / f"{name}.png",
            values,
            f"Success rate vs {PERTURBATION_LABELS.get(name, name)}",
            marker_index=None,
        )


def write_report(path: Path, summary: dict[str, Any]) -> None:
    """生成Markdown鲁棒性报告。"""
    lines: list[str] = []
    lines.append("# Mug 视觉鲁棒性扰动评测报告")
    lines.append("")
    lines.append(f"- 完成条件数：`{summary['condition_count']}`")
    lines.append(f"- 基线（original 无扰动）成功率：`{summary['baseline_rate']:.3f}` "
                 f"(95% CI [{summary['baseline_ci_low']:.3f}, {summary['baseline_ci_high']:.3f}])")
    lines.append("")
    lines.append("## 崩溃阈值")
    lines.append("")
    lines.append("阈值规则：成功率首次跌破 `max(0.5, baseline - drop_pp)` 的最低强度档。")
    lines.append("")
    if summary["pixel_collapse"]:
        lines.append("| 扰动维度 | 崩溃强度 |")
        lines.append("| --- | --- |")
        for name, info in summary["pixel_collapse"].items():
            lines.append(f"| {name} | {info['collapse_intensity']} |")
    if summary.get("combo_collapse"):
        lines.append("| 外观@光照组合 | 崩溃说明 |")
        lines.append("| --- | --- |")
        for label, info in summary["combo_collapse"].items():
            lines.append(f"| {label} | {info} |")
    if not summary["pixel_collapse"] and not summary.get("combo_collapse"):
        lines.append("无扰动维度完成。")
    lines.append("")
    lines.append("## 逐维度成功率（含 Bootstrap 95% CI）")
    lines.append("")
    lines.append("| 扰动 | 强度 | 成功数 | 总数 | 成功率 | CI 低 | CI 高 | 崩溃 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in summary["aggregate_rows"]:
        lines.append(
            f"| {row['perturbation']} | {row['intensity']} | {row['successes']} "
            f"| {row['rollouts']} | {row['success_rate']:.3f} | {row['ci_low']:.3f} "
            f"| {row['ci_high']:.3f} | {row['collapsed']} |"
        )
    lines.append("")
    lines.append("## 失败阶段分布（highest_direct_stage）")
    lines.append("")
    if summary["stage_summary"]:
        lines.append("| 条件 | S1 | S2 | S3 | S4 | S5 | none |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for label in sorted(summary["stage_summary"]):
            counts = summary["stage_summary"][label]
            lines.append(
                f"| {label} | {counts.get('S1', 0)} | {counts.get('S2', 0)} "
                f"| {counts.get('S3', 0)} | {counts.get('S4', 0)} | {counts.get('S5', 0)} "
                f"| {counts.get('none', 0)} |"
            )
    else:
        lines.append("无阶段数据。")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """解析参数并执行鲁棒性扰动评测。"""
    args = build_parser().parse_args(argv)
    config = load_yaml_config(resolve_path(args.config))
    settings = parse_config(config)
    output_dir = resolve_path(args.output_dir or settings["output_dir"])
    perturbation_filter = (
        set(part.strip() for part in args.perturbations.split(",") if part.strip())
        if args.perturbations
        else None
    )
    summary = run_diagnostic(
        args.checkpoint,
        config,
        settings,
        output_dir,
        args.device,
        args.max_scenes,
        perturbation_filter,
        args.skip_appearance,
    )
    print(json.dumps(summary, ensure_ascii=False, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
