"""固定视觉、状态与Flow Matching噪声，诊断语言条件向动作的传播。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from collector.task_spec import TASKS
from evaluate.common import convert_policy_action, find_pretrained_model, load_yaml_config, resolve_path, write_json
from evaluate.rollout import build_prompt, load_policy_bundle, make_policy_observation, set_policy_seed
from sim.environment import CleanTabletopEnv


os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


CUBE_COLORS = ("red", "green")
PAD_COLORS = ("blue", "yellow")
PROMPT_TYPES = ("canonical", "synonym", "unseen")
TASK_IDS = {
    (task.cube_color, task.pad_color): task_id for task_id, task in TASKS.items()
}
REPEAT_ATOL = 1e-6
REPEAT_RTOL = 1e-5


@dataclass(frozen=True)
class ConditioningSpec:
    """一条固定场景语言条件。

    Attributes:
        scene_seed: MuJoCo场景种子。
        cube_color: 指令中的源积木颜色。
        pad_color: 指令中的目标底板颜色。
        prompt_type: 指令措辞类型。
        policy_seed: 固定Flow Matching噪声使用的种子。
    """

    scene_seed: int
    cube_color: str
    pad_color: str
    prompt_type: str
    policy_seed: int

    @property
    def task_id(self) -> str:
        """返回对应的四类任务标识。"""
        return TASK_IDS[(self.cube_color, self.pad_color)]

    @property
    def key(self) -> str:
        """返回稳定且唯一的条件键。"""
        return (
            f"scene_{self.scene_seed}_{self.task_id}_{self.prompt_type}_"
            f"policy_{self.policy_seed}"
        )


def build_parser() -> argparse.ArgumentParser:
    """创建条件传播诊断命令行解析器。"""
    parser = argparse.ArgumentParser(description="诊断红绿语言条件在SmolVLA中的传播")
    parser.add_argument("--checkpoint", type=Path, required=True, help="模型、checkpoint或训练输出目录")
    parser.add_argument("--config", type=Path, required=True, help="诊断YAML配置")
    parser.add_argument("--output-dir", type=Path, required=True, help="独立诊断产物目录")
    parser.add_argument("--device", default="cuda", help="推理设备，默认cuda；可用cpu调试")
    parser.add_argument("--max-scenes", type=int, help="仅运行前N个场景，用于真实模型冒烟")
    parser.add_argument("--prompt-types", nargs="+", choices=PROMPT_TYPES, help="临时覆盖配置中的措辞")
    return parser


def parse_diagnostic_config(config: dict[str, Any]) -> dict[str, Any]:
    """读取并校验conditioning诊断配置。

    Args:
        config: YAML根节点。

    Returns:
        规范化后的诊断配置。
    """
    section = config.get("diagnostic")
    if not isinstance(section, dict):
        raise ValueError("配置必须包含diagnostic映射")
    scene_seeds = [int(value) for value in section.get("scene_seeds", [])]
    prompt_types = [str(value) for value in section.get("prompt_types", [])]
    policy_seed = int(section.get("policy_seed", 20260))
    if not scene_seeds or len(scene_seeds) != len(set(scene_seeds)):
        raise ValueError("scene_seeds必须非空且不重复")
    if not prompt_types or any(value not in PROMPT_TYPES for value in prompt_types):
        raise ValueError(f"prompt_types仅支持{PROMPT_TYPES}")
    if policy_seed != 20260:
        raise ValueError("本实验锁定唯一policy_seed=20260")
    return {
        "scene_seeds": scene_seeds,
        "prompt_types": prompt_types,
        "policy_seed": policy_seed,
        "analysis_horizons": [10, 50],
    }


def build_condition_specs(settings: dict[str, Any]) -> list[ConditioningSpec]:
    """构造场景、颜色和措辞的完整条件矩阵。

    Args:
        settings: 规范化后的诊断配置。

    Returns:
        稳定排序的条件列表。
    """
    specs = [
        ConditioningSpec(scene_seed, cube_color, pad_color, prompt_type, settings["policy_seed"])
        for scene_seed in settings["scene_seeds"]
        for prompt_type in settings["prompt_types"]
        for cube_color in CUBE_COLORS
        for pad_color in PAD_COLORS
    ]
    if len({spec.key for spec in specs}) != len(specs):
        raise ValueError("诊断条件键不唯一")
    return specs


def build_pair_specs(specs: Iterable[ConditioningSpec]) -> list[dict[str, Any]]:
    """从条件矩阵构造红绿源颜色和蓝黄目标颜色配对。

    Args:
        specs: 完整条件列表。

    Returns:
        两类严格配对的索引记录。
    """
    indexed = {
        (spec.scene_seed, spec.cube_color, spec.pad_color, spec.prompt_type): spec
        for spec in specs
    }
    scenes = sorted({spec.scene_seed for spec in specs})
    prompts = list(dict.fromkeys(spec.prompt_type for spec in specs))
    pairs: list[dict[str, Any]] = []
    for scene_seed in scenes:
        for prompt_type in prompts:
            for pad_color in PAD_COLORS:
                left = indexed[(scene_seed, "red", pad_color, prompt_type)]
                right = indexed[(scene_seed, "green", pad_color, prompt_type)]
                pairs.append(
                    {
                        "pair_type": "cube_color",
                        "scene_seed": scene_seed,
                        "prompt_type": prompt_type,
                        "fixed_color": pad_color,
                        "left_key": left.key,
                        "right_key": right.key,
                    }
                )
            for cube_color in CUBE_COLORS:
                left = indexed[(scene_seed, cube_color, "blue", prompt_type)]
                right = indexed[(scene_seed, cube_color, "yellow", prompt_type)]
                pairs.append(
                    {
                        "pair_type": "pad_color",
                        "scene_seed": scene_seed,
                        "prompt_type": prompt_type,
                        "fixed_color": cube_color,
                        "left_key": left.key,
                        "right_key": right.key,
                    }
                )
    return pairs


def sha256_file(path: Path) -> str:
    """流式计算文件SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_sha256() -> str:
    """计算影响conditioning诊断行为的项目源码身份。"""
    project_root = Path(__file__).resolve().parents[1]
    paths = (
        Path(__file__).resolve(),
        project_root / "evaluate" / "common.py",
        project_root / "evaluate" / "rollout.py",
        project_root / "sim" / "environment.py",
        project_root / "collector" / "task_spec.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(project_root)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def array_sha256(value: Any) -> str:
    """计算包含dtype和shape信息的数组SHA-256。

    Args:
        value: NumPy数组或PyTorch Tensor。

    Returns:
        稳定的小写十六进制哈希。
    """
    if hasattr(value, "detach"):
        value = value.detach().cpu().contiguous().numpy()
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def flatten_distance(left: Any, right: Any) -> dict[str, float]:
    """计算两个同形张量的基础距离。

    Args:
        left: 左侧数组。
        right: 右侧数组。

    Returns:
        MAE、RMSE、L2、cosine distance和相对L2。
    """
    lhs = np.asarray(left, dtype=np.float64)
    rhs = np.asarray(right, dtype=np.float64)
    if lhs.shape != rhs.shape:
        raise ValueError(f"比较数组shape不一致: {lhs.shape} != {rhs.shape}")
    delta = lhs - rhs
    lhs_flat = lhs.reshape(-1)
    rhs_flat = rhs.reshape(-1)
    lhs_norm = float(np.linalg.norm(lhs_flat))
    rhs_norm = float(np.linalg.norm(rhs_flat))
    denominator = lhs_norm * rhs_norm
    cosine = 0.0 if denominator == 0.0 and np.array_equal(lhs, rhs) else (
        1.0 if denominator == 0.0 else 1.0 - float(np.dot(lhs_flat, rhs_flat) / denominator)
    )
    return {
        "mae": float(np.mean(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(np.square(delta)))),
        "l2": float(np.linalg.norm(delta.reshape(-1))),
        "cosine_distance": cosine,
        "relative_l2": float(np.linalg.norm(delta.reshape(-1)) / (0.5 * (lhs_norm + rhs_norm) + 1e-12)),
    }


def action_distance(left: np.ndarray, right: np.ndarray, horizon: int) -> dict[str, Any]:
    """统计指定时间窗口内的七维动作差异。

    Args:
        left: 左侧 ``(T, 7)`` 动作。
        right: 右侧 ``(T, 7)`` 动作。
        horizon: 比较前多少步。

    Returns:
        总体、逐时间步、逐维和机械臂/夹爪指标。
    """
    lhs = np.asarray(left, dtype=np.float64)
    rhs = np.asarray(right, dtype=np.float64)
    if lhs.shape != rhs.shape or lhs.ndim != 2 or lhs.shape[1] != 7:
        raise ValueError(f"action chunk必须同为(T, 7)，实际{lhs.shape}与{rhs.shape}")
    if horizon <= 0 or horizon > lhs.shape[0]:
        raise ValueError(f"horizon超出chunk范围: {horizon}/{lhs.shape[0]}")
    lhs = lhs[:horizon]
    rhs = rhs[:horizon]
    delta = lhs - rhs
    result: dict[str, Any] = flatten_distance(lhs, rhs)
    result.update(
        {
            "horizon": horizon,
            "per_step_l2": np.linalg.norm(delta, axis=1).tolist(),
            "per_dimension_mae": np.mean(np.abs(delta), axis=0).tolist(),
            "per_dimension_max_abs": np.max(np.abs(delta), axis=0).tolist(),
            "arm_mae": float(np.mean(np.abs(delta[:, :6]))),
            "gripper_mae": float(np.mean(np.abs(delta[:, 6]))),
        }
    )
    return result


def clip_action_chunk(chunk: np.ndarray, arm_ctrlrange: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """逐步限位物理动作chunk。

    Args:
        chunk: ``(T, 7)``物理动作。
        arm_ctrlrange: 六关节控制范围。

    Returns:
        裁剪后动作和 ``(T, 7)`` 裁剪掩码。
    """
    commands = []
    masks = []
    for action in np.asarray(chunk):
        safe = convert_policy_action(action, arm_ctrlrange)
        commands.append(safe.command)
        masks.append(safe.clipped_mask)
    return np.asarray(commands, dtype=np.float32), np.asarray(masks, dtype=np.bool_)


def repeat_is_deterministic(first: np.ndarray, second: np.ndarray) -> bool:
    """按锁定容差判断完全相同输入的重复推理。"""
    return bool(np.allclose(first, second, atol=REPEAT_ATOL, rtol=REPEAT_RTOL))


def diagnostic_hint(
    feature_distance: float,
    physical_distance: float,
    clipped_distance: float,
    repeat_distance: float,
) -> str:
    """根据相对底噪生成定位提示，不作为模型效果通过标准。

    Args:
        feature_distance: VLM/KV特征差异。
        physical_distance: 物理动作差异。
        clipped_distance: 裁剪后动作差异。
        repeat_distance: 同输入重复推理底噪。

    Returns:
        分层诊断提示。
    """
    floor = max(10.0 * repeat_distance, 1e-8)
    if feature_distance <= floor and physical_distance <= floor:
        return "vlm_and_action_insensitive"
    if feature_distance > floor and physical_distance <= floor:
        return "vlm_sensitive_action_insensitive"
    if physical_distance > floor and clipped_distance <= 0.1 * physical_distance:
        return "action_difference_collapsed_by_clipping"
    return "feature_and_action_differences_present"


def processed_input_hashes(processed: dict[str, Any]) -> dict[str, str]:
    """计算预处理后视觉和状态张量哈希，主动排除语言张量。"""
    keys = sorted(
        key for key in processed if key.startswith("observation.images.") or key == "observation.state"
    )
    return {key: array_sha256(processed[key]) for key in keys}


def extract_language_features(policy: Any, processed: dict[str, Any]) -> dict[str, np.ndarray]:
    """沿真实SmolVLA prefix路径提取语言特征和逐层KV。

    Args:
        policy: 已加载SmolVLA策略。
        processed: checkpoint预处理后的单条观测。

    Returns:
        仅包含有效语言区间的输入embedding、最终特征和KV数组。
    """
    import torch
    from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    if int(getattr(policy.config, "prefix_length", 0)) != 0:
        raise ValueError("当前诊断要求checkpoint的prefix_length=0，以精确定位语言区间")
    images, img_masks = policy.prepare_images(processed)
    state = policy.prepare_state(processed)
    lang_tokens = processed[OBS_LANGUAGE_TOKENS]
    lang_masks = processed[OBS_LANGUAGE_ATTENTION_MASK].bool()
    model = policy.model
    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
        images, img_masks, lang_tokens, lang_masks, state=state
    )
    state_emb = model.state_proj(state)
    state_length = 1 if state_emb.ndim == 2 else int(state_emb.shape[1])
    language_length = int(lang_tokens.shape[1])
    language_start = int(prefix_embs.shape[1]) - state_length - language_length
    language_end = language_start + language_length
    if language_start < 0 or language_end > prefix_embs.shape[1]:
        raise ValueError("无法从真实prefix长度定位语言区间")

    raw = model.vlm_with_expert.embed_language_tokens(lang_tokens)
    raw = raw * float(raw.shape[-1] ** 0.5)
    attention_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    outputs, cache = model.vlm_with_expert.forward(
        attention_mask=attention_2d,
        position_ids=position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=bool(policy.config.use_cache),
        fill_kv_cache=True,
    )
    active_indices = torch.nonzero(lang_masks[0], as_tuple=False).flatten()
    if active_indices.numel() == 0:
        raise ValueError("语言attention mask没有有效Token")
    prefix_indices = active_indices + language_start
    result = {
        "token_ids": lang_tokens[0, active_indices].detach().cpu().numpy().astype(np.int64),
        "active_token_indices": active_indices.detach().cpu().numpy().astype(np.int64),
        "prefix_token_indices": prefix_indices.detach().cpu().numpy().astype(np.int64),
        "raw_embedding": raw[0, active_indices].detach().float().cpu().numpy(),
        "final_hidden": outputs[0][0, prefix_indices].detach().float().cpu().numpy(),
    }
    for layer_index in sorted(cache):
        layer_cache = cache[layer_index]
        result[f"layer_{layer_index:02d}_key"] = (
            layer_cache["key_states"][0, prefix_indices].detach().float().cpu().numpy()
        )
        result[f"layer_{layer_index:02d}_value"] = (
            layer_cache["value_states"][0, prefix_indices].detach().float().cpu().numpy()
        )
    return result


def infer_action_chunk(
    policy: Any,
    postprocessor: Any,
    processed: dict[str, Any],
    noise: Any,
    arm_ctrlrange: np.ndarray,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """使用官方接口生成归一化、物理和裁剪动作。

    Args:
        policy: SmolVLA策略。
        postprocessor: checkpoint动作后处理器。
        processed: 预处理观测。
        noise: 固定Flow Matching噪声。
        arm_ctrlrange: 六关节控制范围。
        device: 推理设备。

    Returns:
        归一化动作、物理动作、裁剪动作和裁剪掩码。
    """
    import torch

    policy.reset()
    autocast = (
        torch.autocast(device_type="cuda")
        if device.startswith("cuda") and bool(getattr(policy.config, "use_amp", False))
        else nullcontext()
    )
    with torch.inference_mode(), autocast:
        normalized_tensor = policy.predict_action_chunk(processed, noise=noise.clone())
        physical_tensor = postprocessor(normalized_tensor)
    normalized = normalized_tensor[0].detach().float().cpu().numpy()
    if hasattr(physical_tensor, "detach"):
        physical = physical_tensor[0].detach().float().cpu().numpy()
    else:
        physical = np.asarray(physical_tensor)[0]
    clipped, clipped_mask = clip_action_chunk(physical, arm_ctrlrange)
    return normalized, physical.astype(np.float32), clipped, clipped_mask


def compare_feature_sets(
    pair: dict[str, Any],
    features: dict[str, dict[str, np.ndarray]],
) -> list[dict[str, Any]]:
    """生成一组条件在各类语言特征上的距离记录。"""
    left = features[pair["left_key"]]
    right = features[pair["right_key"]]
    rows = []
    left_tokens = np.asarray(left["token_ids"])
    right_tokens = np.asarray(right["token_ids"])
    if left_tokens.shape != right_tokens.shape:
        raise ValueError("严格颜色配对的有效Token长度必须一致")
    changed_positions = np.flatnonzero(left_tokens != right_tokens)
    if changed_positions.size == 0:
        raise ValueError("颜色配对没有找到差异Token位置")
    feature_names = sorted(set(left) & set(right) - {"token_ids", "active_token_indices", "prefix_token_indices"})
    for feature_name in feature_names:
        metrics = flatten_distance(left[feature_name], right[feature_name])
        lhs = np.asarray(left[feature_name])
        rhs = np.asarray(right[feature_name])
        per_token_l2 = np.linalg.norm((lhs - rhs).reshape(lhs.shape[0], -1), axis=1)
        changed_metrics = flatten_distance(lhs[changed_positions], rhs[changed_positions])
        rows.append(
            {
                **pair,
                "feature": feature_name,
                "changed_token_positions": json.dumps(changed_positions.tolist()),
                "changed_token_l2": changed_metrics["l2"],
                "per_token_l2_mean": float(np.mean(per_token_l2)),
                "per_token_l2_max": float(np.max(per_token_l2)),
                **metrics,
            }
        )
    return rows


def compare_action_sets(
    pair: dict[str, Any],
    actions: dict[str, dict[str, np.ndarray]],
    horizons: Iterable[int] = (10, 50),
) -> list[dict[str, Any]]:
    """生成一组条件在三种动作阶段上的距离记录。"""
    rows = []
    left = actions[pair["left_key"]]
    right = actions[pair["right_key"]]
    for stage in ("normalized", "physical", "clipped"):
        for horizon in horizons:
            metrics = action_distance(left[stage], right[stage], horizon)
            row = {**pair, "stage": stage, **metrics}
            for key in ("per_step_l2", "per_dimension_mae", "per_dimension_max_abs"):
                row[key] = json.dumps(row[key], ensure_ascii=False)
            rows.append(row)
    for horizon in horizons:
        physical_delta = np.abs(left["physical"][:horizon] - right["physical"][:horizon])
        clipped_delta = np.abs(left["clipped"][:horizon] - right["clipped"][:horizon])
        changed_before = physical_delta > 1e-7
        equal_after = clipped_delta <= 1e-7
        collapsed_elements = changed_before & equal_after
        collapsed_steps = changed_before.any(axis=1) & equal_after.all(axis=1)
        physical_row = next(
            row for row in rows if row["stage"] == "physical" and row["horizon"] == horizon
        )
        clipped_row = next(
            row for row in rows if row["stage"] == "clipped" and row["horizon"] == horizon
        )
        physical_row["clipping_collapsed_element_count"] = int(collapsed_elements.sum())
        physical_row["clipping_collapsed_step_count"] = int(collapsed_steps.sum())
        physical_row["clipped_difference_retained_ratio"] = float(clipped_row["l2"]) / (
            float(physical_row["l2"]) + 1e-12
        )
    for row in rows:
        row.setdefault("clipping_collapsed_element_count", "")
        row.setdefault("clipping_collapsed_step_count", "")
        row.setdefault("clipped_difference_retained_ratio", "")
    return rows


def add_control_ratios(
    feature_rows: list[dict[str, Any]],
    action_rows: list[dict[str, Any]],
) -> None:
    """为每层特征和动作阶段添加红绿相对蓝黄的聚合差异比。"""
    for feature_name in {row["feature"] for row in feature_rows}:
        source = [row["relative_l2"] for row in feature_rows if row["feature"] == feature_name and row["pair_type"] == "cube_color"]
        target = [row["relative_l2"] for row in feature_rows if row["feature"] == feature_name and row["pair_type"] == "pad_color"]
        ratio = float(np.median(source)) / (float(np.median(target)) + 1e-12)
        for row in feature_rows:
            if row["feature"] == feature_name:
                row["source_to_target_relative_l2_ratio"] = ratio
    groups = {(row["stage"], row["horizon"]) for row in action_rows}
    for stage, horizon in groups:
        source = [row["l2"] for row in action_rows if row["stage"] == stage and row["horizon"] == horizon and row["pair_type"] == "cube_color"]
        target = [row["l2"] for row in action_rows if row["stage"] == stage and row["horizon"] == horizon and row["pair_type"] == "pad_color"]
        ratio = float(np.median(source)) / (float(np.median(target)) + 1e-12)
        for row in action_rows:
            if row["stage"] == stage and row["horizon"] == horizon:
                row["source_to_target_l2_ratio"] = ratio


def median_for(rows: list[dict[str, Any]], **filters: Any) -> float:
    """按字段过滤后返回L2中位数，空集合返回零。"""
    values = [
        float(row["l2"])
        for row in rows
        if all(row.get(key) == value for key, value in filters.items())
    ]
    return float(np.median(values)) if values else 0.0


def build_summary(
    specs: list[ConditioningSpec],
    pairs: list[dict[str, Any]],
    records: list[dict[str, Any]],
    feature_rows: list[dict[str, Any]],
    action_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """聚合实验规模、重复有效性与分层诊断提示。"""
    nondeterministic = [record["condition_key"] for record in records if not record["repeat_deterministic"]]
    source_action = median_for(
        action_rows, pair_type="cube_color", stage="physical", horizon=50
    )
    target_action = median_for(
        action_rows, pair_type="pad_color", stage="physical", horizon=50
    )
    source_clipped = median_for(
        action_rows, pair_type="cube_color", stage="clipped", horizon=50
    )
    source_feature = median_for(
        feature_rows, pair_type="cube_color", feature="final_hidden"
    )
    repeat_floor = float(np.median([record["repeat_normalized_l2"] for record in records]))
    clipped_by_dimension = np.sum(
        np.asarray([record["clipped_counts_by_dimension"] for record in records], dtype=np.int64),
        axis=0,
    ).tolist()
    status = "invalid" if nondeterministic else "complete"
    return {
        "status": status,
        "condition_count": len(specs),
        "action_chunk_count": len(specs) * 2,
        "pair_count": len(pairs),
        "cube_color_pair_count": sum(pair["pair_type"] == "cube_color" for pair in pairs),
        "pad_color_pair_count": sum(pair["pair_type"] == "pad_color" for pair in pairs),
        "nondeterministic_repeat_count": len(nondeterministic),
        "nondeterministic_condition_keys": nondeterministic,
        "median_source_color_physical_l2_full50": source_action,
        "median_target_color_physical_l2_full50": target_action,
        "source_to_target_action_ratio": source_action / (target_action + 1e-12),
        "median_source_color_clipped_l2_full50": source_clipped,
        "median_source_color_final_hidden_l2": source_feature,
        "median_repeat_normalized_l2": repeat_floor,
        "clipped_counts_by_dimension": clipped_by_dimension,
        "diagnostic_hint": diagnostic_hint(
            source_feature, source_action, source_clipped, repeat_floor
        ),
        "interpretation_boundary": (
            "特征或action chunk存在差异，只能证明模型对文本变化敏感，不能证明其正确定位并抓取了对应颜色积木。"
        ),
    }


def save_npz(path: Path, groups: dict[str, dict[str, np.ndarray]]) -> None:
    """把按条件分组的数组保存为压缩NPZ。"""
    arrays = {
        f"{condition_key}__{name}": value
        for condition_key, group in groups.items()
        for name, value in group.items()
    }
    np.savez_compressed(path, **arrays)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """写出UTF-8 BOM CSV；空记录只创建空文件。"""
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        if not rows:
            return
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_plots(output_dir: Path, feature_rows: list[dict[str, Any]], action_rows: list[dict[str, Any]]) -> None:
    """生成逐层特征与逐时间步动作差异图。

    优先使用Matplotlib；本机评测环境没有安装该包时使用Pillow绘制简洁折线图，
    避免为了诊断图修改锁定环境。
    """
    kv_rows = [
        row for row in feature_rows
        if row["pair_type"] == "cube_color" and row["feature"].endswith("_key")
    ]
    feature_names = sorted({row["feature"] for row in kv_rows})
    feature_values = [np.median([row["relative_l2"] for row in kv_rows if row["feature"] == name]) for name in feature_names]
    selected = [
        row for row in action_rows
        if row["pair_type"] == "cube_color" and row["stage"] == "physical" and row["horizon"] == 50
    ]
    step_values = np.median(
        np.asarray([json.loads(row["per_step_l2"]) for row in selected], dtype=np.float64), axis=0
    )
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        write_pillow_line_plot(
            output_dir / "feature_delta.png",
            np.asarray(feature_values, dtype=np.float64),
            "VLM KV relative L2",
            marker_index=None,
        )
        write_pillow_line_plot(
            output_dir / "action_delta.png",
            step_values,
            "Red-green physical action L2",
            marker_index=9,
        )
        return

    figure, axis = plt.subplots(figsize=(9, 4.5))
    axis.plot(range(len(feature_names)), feature_values, marker="o")
    axis.set_xlabel("VLM KV layer")
    axis.set_ylabel("Median relative L2 (red vs green)")
    axis.set_title("Language condition propagation through VLM KV cache")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_dir / "feature_delta.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 4.5))
    axis.plot(np.arange(1, len(step_values) + 1), step_values)
    axis.axvline(10, color="tab:red", linestyle="--", label="execution horizon=10")
    axis.set_xlabel("Action chunk step")
    axis.set_ylabel("Median physical action L2")
    axis.set_title("Red-green action chunk difference")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "action_delta.png", dpi=160)
    plt.close(figure)


def write_pillow_line_plot(
    path: Path,
    values: np.ndarray,
    title: str,
    marker_index: int | None,
) -> None:
    """在无Matplotlib环境中用Pillow生成可读折线图。

    Args:
        path: PNG输出路径。
        values: 一维有限数值序列。
        title: 图标题。
        marker_index: 可选的竖线索引，例如前10步边界使用9。
    """
    from PIL import Image, ImageDraw

    sequence = np.asarray(values, dtype=np.float64).reshape(-1)
    if sequence.size == 0 or not np.isfinite(sequence).all():
        raise ValueError("Pillow折线图需要非空有限数值")
    width, height = 1000, 500
    left, right, top, bottom = 85, 35, 65, 65
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((left, 20), title, fill="black")
    draw.line((left, top, left, height - bottom), fill="black", width=2)
    draw.line((left, height - bottom, width - right, height - bottom), fill="black", width=2)
    lower = float(np.min(sequence))
    upper = float(np.max(sequence))
    span = upper - lower if upper > lower else 1.0
    x_span = width - left - right
    y_span = height - top - bottom
    points = []
    for index, value in enumerate(sequence):
        ratio = index / max(1, sequence.size - 1)
        x = left + int(round(ratio * x_span))
        y = height - bottom - int(round((float(value) - lower) / span * y_span))
        points.append((x, y))
    if len(points) > 1:
        draw.line(points, fill=(31, 119, 180), width=3)
    else:
        x, y = points[0]
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(31, 119, 180))
    if marker_index is not None and 0 <= marker_index < sequence.size:
        marker_x = points[marker_index][0]
        draw.line((marker_x, top, marker_x, height - bottom), fill=(214, 39, 40), width=2)
    draw.text((10, top), f"max={upper:.5g}", fill="black")
    draw.text((10, height - bottom - 12), f"min={lower:.5g}", fill="black")
    draw.text((left, height - bottom + 20), "1", fill="black")
    draw.text((width - right - 30, height - bottom + 20), str(sequence.size), fill="black")
    image.save(path, format="PNG")


def write_report(path: Path, summary: dict[str, Any]) -> None:
    """写出中文人工可读诊断报告。"""
    lines = [
        "# 红绿指令VLM特征与Action Chunk对照实验",
        "",
        f"- 运行状态：`{summary['status']}`",
        f"- 条件数：{summary['condition_count']}",
        f"- Action Chunk数：{summary['action_chunk_count']}",
        f"- 红绿源颜色配对：{summary['cube_color_pair_count']}",
        f"- 蓝黄目标颜色对照：{summary['pad_color_pair_count']}",
        f"- 非确定性重复：{summary['nondeterministic_repeat_count']}",
        "",
        "## 聚合结果",
        "",
        f"- 红绿最终语言特征L2中位数：`{summary['median_source_color_final_hidden_l2']:.8g}`",
        f"- 红绿物理动作L2中位数（50步）：`{summary['median_source_color_physical_l2_full50']:.8g}`",
        f"- 蓝黄物理动作L2中位数（50步）：`{summary['median_target_color_physical_l2_full50']:.8g}`",
        f"- 源颜色/目标颜色动作差异比：`{summary['source_to_target_action_ratio']:.8g}`",
        f"- 红绿裁剪后动作L2中位数（50步）：`{summary['median_source_color_clipped_l2_full50']:.8g}`",
        f"- 同输入重复推理底噪L2中位数：`{summary['median_repeat_normalized_l2']:.8g}`",
        f"- 分层诊断提示：`{summary['diagnostic_hint']}`",
        "",
        "## 解读边界",
        "",
        summary["interpretation_boundary"],
        "初始chunk相似也不能单独证明模型忽略颜色，因为前几步可能是共享的接近动作；应同时查看前10步和完整50步。",
        "若特征不同而动作相近，优先检查动作专家是否使用语言条件；若动作不同但闭环位置相近，再结合物体位置与闭环轨迹诊断视觉落点。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def git_identity() -> dict[str, Any]:
    """读取Git提交与工作区状态。"""
    try:
        project_root = Path(__file__).resolve().parents[1]
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project_root, text=True, encoding="utf-8"
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=project_root, text=True, encoding="utf-8"
            ).strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unavailable", "dirty": None}


def build_manifest(
    checkpoint: Path,
    config: dict[str, Any],
    settings: dict[str, Any],
    device: str,
    noise_sha256: str,
) -> dict[str, Any]:
    """构造checkpoint、配置、环境和固定噪声运行清单。"""
    import mujoco
    import torch

    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint / "model.safetensors"),
        "checkpoint_config_sha256": sha256_file(checkpoint / "config.json"),
        "source_sha256": source_sha256(),
        "preprocessor_sha256": sha256_file(checkpoint / "policy_preprocessor.json"),
        "postprocessor_sha256": sha256_file(checkpoint / "policy_postprocessor.json"),
        "git": git_identity(),
        "config": config,
        "settings": settings,
        "fixed_noise": {"shape": [1, 50, 32], "policy_seed": 20260, "sha256": noise_sha256},
        "repeat_tolerance": {"atol": REPEAT_ATOL, "rtol": REPEAT_RTOL},
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "device": device,
            "lerobot": importlib.metadata.version("lerobot"),
            "mujoco": mujoco.__version__,
        },
        "amp_enabled": bool(json.loads((checkpoint / "config.json").read_text(encoding="utf-8")).get("use_amp")),
        "mujoco_stepped": False,
        "closed_loop_success_measured": False,
    }


def run_diagnostic(
    checkpoint: Path,
    config: dict[str, Any],
    settings: dict[str, Any],
    output_dir: Path,
    device: str,
) -> dict[str, Any]:
    """执行完整固定条件诊断并写出全部产物。"""
    import imageio.v2 as imageio
    import torch

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("请求cuda诊断，但当前环境没有可用CUDA")
    specs = build_condition_specs(settings)
    pairs = build_pair_specs(specs)
    policy, preprocessor, postprocessor = load_policy_bundle(checkpoint, device, execution_horizon=50)
    policy.eval()
    generator = torch.Generator(device=device)
    generator.manual_seed(settings["policy_seed"])
    fixed_noise = torch.randn(
        (1, int(policy.config.chunk_size), int(policy.config.max_action_dim)),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    if tuple(fixed_noise.shape) != (1, 50, 32):
        raise ValueError(f"checkpoint噪声shape不是锁定的(1, 50, 32): {tuple(fixed_noise.shape)}")
    noise_cpu = fixed_noise.detach().cpu().numpy()
    np.save(output_dir / "fixed_noise.npy", noise_cpu)
    noise_hash = array_sha256(noise_cpu)
    write_json(output_dir / "run_manifest.json", build_manifest(checkpoint, config, settings, device, noise_hash))

    fixed_inputs_dir = output_dir / "fixed_inputs"
    fixed_inputs_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    features: dict[str, dict[str, np.ndarray]] = {}
    actions: dict[str, dict[str, np.ndarray]] = {}
    baseline_processed_hashes: dict[int, dict[str, str]] = {}
    arm_ctrlrange: np.ndarray | None = None

    with (output_dir / "condition_records.jsonl").open("w", encoding="utf-8", newline="\n") as jsonl:
        with CleanTabletopEnv() as env:
            arm_ctrlrange = env.model.actuator_ctrlrange[:6].copy()
            for scene_seed in settings["scene_seeds"]:
                snapshot = env.reset(scene_seed)
                images = env.capture_training_images()
                state = env.get_state()
                layout = env.task_layout()
                scene_prefix = fixed_inputs_dir / f"scene_{scene_seed}"
                imageio.imwrite(scene_prefix.with_name(f"scene_{scene_seed}_agent.png"), images["agent"])
                imageio.imwrite(scene_prefix.with_name(f"scene_{scene_seed}_wrist.png"), images["wrist"])
                scene_record = {
                    "scene_seed": scene_seed,
                    "state": state.astype(float).tolist(),
                    "state_sha256": array_sha256(state),
                    "agent_sha256": array_sha256(images["agent"]),
                    "wrist_sha256": array_sha256(images["wrist"]),
                    "cube_initial_poses": snapshot.cube_initial_poses.tolist(),
                    "pad_positions": snapshot.pad_positions.tolist(),
                    "layout_positions": {
                        name: np.asarray(value["position"], dtype=float).tolist() for name, value in layout.items()
                    },
                }
                write_json(scene_prefix.with_suffix(".json"), scene_record)
                scene_specs = [spec for spec in specs if spec.scene_seed == scene_seed]
                for spec in scene_specs:
                    set_policy_seed(spec.policy_seed)
                    prompt = build_prompt(spec.task_id, spec.prompt_type)
                    processed = preprocessor(make_policy_observation(images, state, prompt))
                    current_hashes = processed_input_hashes(processed)
                    baseline = baseline_processed_hashes.setdefault(scene_seed, current_hashes)
                    if current_hashes != baseline:
                        raise RuntimeError(f"scene={scene_seed}的预处理视觉或状态随指令发生变化")
                    with torch.inference_mode(), (
                        torch.autocast(device_type="cuda")
                        if device.startswith("cuda") and bool(policy.config.use_amp)
                        else nullcontext()
                    ):
                        features[spec.key] = extract_language_features(policy, processed)
                    first = infer_action_chunk(
                        policy, postprocessor, processed, fixed_noise, arm_ctrlrange, device
                    )
                    second = infer_action_chunk(
                        policy, postprocessor, processed, fixed_noise, arm_ctrlrange, device
                    )
                    repeat_metrics = flatten_distance(first[0], second[0])
                    deterministic = repeat_is_deterministic(first[0], second[0])
                    actions[spec.key] = {
                        "normalized": first[0],
                        "physical": first[1],
                        "clipped": first[2],
                        "clipped_mask": first[3],
                        "repeat_normalized": second[0],
                        "repeat_physical": second[1],
                        "repeat_clipped": second[2],
                    }
                    record = {
                        "condition_key": spec.key,
                        "scene_seed": spec.scene_seed,
                        "policy_seed": spec.policy_seed,
                        "task_id": spec.task_id,
                        "cube_color": spec.cube_color,
                        "pad_color": spec.pad_color,
                        "prompt_type": spec.prompt_type,
                        "prompt": prompt,
                        "noise_sha256": noise_hash,
                        "processed_input_hashes": current_hashes,
                        "active_token_ids": features[spec.key]["token_ids"].tolist(),
                        "active_token_indices": features[spec.key]["active_token_indices"].tolist(),
                        "prefix_token_indices": features[spec.key]["prefix_token_indices"].tolist(),
                        "repeat_deterministic": deterministic,
                        "repeat_normalized_l2": repeat_metrics["l2"],
                        "repeat_normalized_max_abs": float(np.max(np.abs(first[0] - second[0]))),
                        "clipped_value_count": int(first[3].sum()),
                        "clipped_counts_by_dimension": first[3].sum(axis=0).astype(int).tolist(),
                    }
                    records.append(record)
                    jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")
                    jsonl.flush()

    feature_rows = [row for pair in pairs for row in compare_feature_sets(pair, features)]
    action_rows = [row for pair in pairs for row in compare_action_sets(pair, actions)]
    add_control_ratios(feature_rows, action_rows)
    save_npz(output_dir / "language_features.npz", features)
    save_npz(output_dir / "action_chunks.npz", actions)
    write_csv(output_dir / "feature_comparison.csv", feature_rows)
    write_csv(output_dir / "action_comparison.csv", action_rows)
    summary = build_summary(specs, pairs, records, feature_rows, action_rows)
    write_json(output_dir / "summary.json", summary)
    write_report(output_dir / "report.md", summary)
    write_plots(output_dir, feature_rows, action_rows)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    """解析参数、运行诊断并返回有效性状态。"""
    args = build_parser().parse_args(argv)
    checkpoint = find_pretrained_model(args.checkpoint)
    config_path = resolve_path(args.config)
    config = load_yaml_config(config_path)
    settings = parse_diagnostic_config(config)
    if args.prompt_types:
        settings["prompt_types"] = list(args.prompt_types)
    if args.max_scenes is not None:
        if args.max_scenes <= 0:
            raise ValueError("max-scenes必须大于零")
        settings["scene_seeds"] = settings["scene_seeds"][: args.max_scenes]
    output_dir = resolve_path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"诊断输出目录已存在且非空，请更换目录: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = run_diagnostic(checkpoint, config, settings, output_dir, args.device)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
