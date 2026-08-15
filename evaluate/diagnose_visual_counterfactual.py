"""通过MuJoCo视觉反事实诊断SmolVLA颜色空间绑定链路。"""

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
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

from evaluate.common import find_pretrained_model, load_yaml_config, resolve_path, write_json
from evaluate.diagnose_conditioning import (
    REPEAT_ATOL,
    REPEAT_RTOL,
    action_distance,
    array_sha256,
    flatten_distance,
    infer_action_chunk,
    repeat_is_deterministic,
    save_npz,
    sha256_file,
    write_pillow_line_plot,
)
from evaluate.rollout import build_prompt, load_policy_bundle, make_policy_observation, set_policy_seed
from sim.environment import (
    ARM_JOINT_NAMES,
    TASK_CUBE_BODY_NAMES,
    TASK_CUBE_GEOM_NAMES,
    CleanTabletopEnv,
)


os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


CAMERAS = {"agent": "agentview", "wrist": "d435i_rgb"}
CUBE_COLORS = ("red", "green")
VISUAL_VARIANTS = ("original", "swap_positions", "neutralize_red", "neutralize_green")
TOKEN_GRID_SIZE = 8
IMAGE_TOKEN_COUNT = 64
BOOTSTRAP_SAMPLES = 10_000
PIXEL_CHANGE_THRESHOLD = 0
OUTSIDE_ROI_DRIFT_LIMIT = 0.02


@dataclass(frozen=True)
class VisualCondition:
    """一条语言—视觉反事实条件。

    Attributes:
        scene_seed: MuJoCo场景种子。
        source_color: 指令要求抓取的积木颜色。
        pad_color: 固定目标底板颜色。
        prompt_type: 固定语言措辞。
        visual_variant: 当前视觉反事实版本。
        policy_seed: 唯一Flow Matching噪声种子。
    """

    scene_seed: int
    source_color: str
    pad_color: str
    prompt_type: str
    visual_variant: str
    policy_seed: int

    @property
    def task_id(self) -> str:
        """返回当前颜色组合对应任务标识。"""
        return f"{self.source_color}_on_{self.pad_color}"

    @property
    def key(self) -> str:
        """返回稳定且唯一的反事实条件键。"""
        return (
            f"scene_{self.scene_seed}_{self.task_id}_{self.prompt_type}_"
            f"{self.visual_variant}_policy_{self.policy_seed}"
        )


@dataclass(frozen=True)
class SceneBaseState:
    """视觉干预前必须完整恢复的MuJoCo场景字段。"""

    cube_qpos: np.ndarray
    cube_rgba: np.ndarray
    robot_state: np.ndarray


def build_parser() -> argparse.ArgumentParser:
    """创建视觉反事实诊断命令行解析器。"""
    parser = argparse.ArgumentParser(description="诊断SmolVLA红绿视觉—语言空间绑定")
    parser.add_argument("--checkpoint", type=Path, required=True, help="模型、checkpoint或训练输出目录")
    parser.add_argument("--config", type=Path, required=True, help="视觉反事实YAML配置")
    parser.add_argument("--output-dir", type=Path, required=True, help="独立诊断产物目录")
    parser.add_argument("--device", default="cuda", help="推理设备，默认cuda")
    parser.add_argument("--max-scenes", type=int, help="仅运行前N个场景，用于真实模型冒烟")
    return parser


def parse_config(config: dict[str, Any]) -> dict[str, Any]:
    """读取并校验视觉反事实诊断配置。

    Args:
        config: YAML根节点。

    Returns:
        规范化配置。
    """
    section = config.get("diagnostic")
    if not isinstance(section, dict):
        raise ValueError("配置必须包含diagnostic映射")
    scene_seeds = [int(value) for value in section.get("scene_seeds", [])]
    source_colors = [str(value) for value in section.get("source_colors", [])]
    visual_variants = [str(value) for value in section.get("visual_variants", [])]
    neutral_rgba = np.asarray(section.get("neutral_rgba", []), dtype=np.float64)
    horizons = [int(value) for value in section.get("analysis_horizons", [])]
    policy_seed = int(section.get("policy_seed", -1))
    if not scene_seeds or len(scene_seeds) != len(set(scene_seeds)):
        raise ValueError("scene_seeds必须非空且不重复")
    if source_colors != list(CUBE_COLORS):
        raise ValueError(f"source_colors必须锁定为{list(CUBE_COLORS)}")
    if visual_variants != list(VISUAL_VARIANTS):
        raise ValueError(f"visual_variants必须锁定为{list(VISUAL_VARIANTS)}")
    if section.get("pad_color") != "blue" or section.get("prompt_type") != "canonical":
        raise ValueError("首轮实验锁定blue目标和canonical措辞")
    if neutral_rgba.shape != (4,) or not np.isfinite(neutral_rgba).all():
        raise ValueError("neutral_rgba必须是有限四维RGBA")
    if np.any(neutral_rgba < 0.0) or np.any(neutral_rgba > 1.0):
        raise ValueError("neutral_rgba必须位于[0,1]")
    if horizons != [10, 50]:
        raise ValueError("analysis_horizons必须锁定为[10, 50]")
    if policy_seed != 20260:
        raise ValueError("本实验只允许policy_seed=20260")
    return {
        "scene_seeds": scene_seeds,
        "source_colors": source_colors,
        "pad_color": "blue",
        "prompt_type": "canonical",
        "visual_variants": visual_variants,
        "neutral_rgba": neutral_rgba.tolist(),
        "policy_seed": policy_seed,
        "analysis_horizons": horizons,
    }


def build_conditions(settings: dict[str, Any]) -> list[VisualCondition]:
    """构造48个语言—视觉唯一条件。"""
    conditions = [
        VisualCondition(
            scene_seed=scene_seed,
            source_color=source_color,
            pad_color=settings["pad_color"],
            prompt_type=settings["prompt_type"],
            visual_variant=variant,
            policy_seed=settings["policy_seed"],
        )
        for scene_seed in settings["scene_seeds"]
        for source_color in settings["source_colors"]
        for variant in settings["visual_variants"]
    ]
    if len({condition.key for condition in conditions}) != len(conditions):
        raise ValueError("视觉反事实条件键不唯一")
    return conditions


def repeat_control_keys(conditions: Sequence[VisualCondition]) -> list[str]:
    """返回每个scene和颜色的12个original重复控制键。"""
    return [condition.key for condition in conditions if condition.visual_variant == "original"]


def _object_id(model: Any, object_type: Any, name: str) -> int:
    """读取MuJoCo对象ID并拒绝缺失对象。"""
    import mujoco

    object_id = int(mujoco.mj_name2id(model, object_type, name))
    if object_id < 0:
        raise ValueError(f"MuJoCo缺少对象: {name}")
    return object_id


def capture_scene_base(env: CleanTabletopEnv) -> SceneBaseState:
    """捕获两块积木qpos、RGBA和固定机器人状态。"""
    import mujoco

    qposes = []
    rgba = []
    for joint_name, geom_name in zip(
        ("task_red_cube_free_joint", "task_green_cube_free_joint"),
        TASK_CUBE_GEOM_NAMES,
        strict=True,
    ):
        joint_id = _object_id(env.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        address = int(env.model.jnt_qposadr[joint_id])
        qposes.append(env.data.qpos[address:address + 7].copy())
        geom_id = _object_id(env.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        rgba.append(env.model.geom_rgba[geom_id].copy())
    return SceneBaseState(
        cube_qpos=np.asarray(qposes, dtype=np.float64),
        cube_rgba=np.asarray(rgba, dtype=np.float64),
        robot_state=env.get_state().copy(),
    )


def restore_scene_base(env: CleanTabletopEnv, base: SceneBaseState) -> None:
    """恢复积木qpos和RGBA并执行一次纯运动学更新。"""
    import mujoco

    for index, (joint_name, geom_name) in enumerate(
        zip(
            ("task_red_cube_free_joint", "task_green_cube_free_joint"),
            TASK_CUBE_GEOM_NAMES,
            strict=True,
        )
    ):
        joint_id = _object_id(env.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        address = int(env.model.jnt_qposadr[joint_id])
        env.data.qpos[address:address + 7] = base.cube_qpos[index]
        geom_id = _object_id(env.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        env.model.geom_rgba[geom_id] = base.cube_rgba[index]
    mujoco.mj_forward(env.model, env.data)


@contextmanager
def visual_variant_context(
    env: CleanTabletopEnv,
    base: SceneBaseState,
    variant: str,
    neutral_rgba: Sequence[float],
) -> Iterator[None]:
    """临时应用一种视觉反事实并保证退出后恢复。

    Args:
        env: 当前MuJoCo环境。
        base: 当前scene原始状态。
        variant: 四种锁定视觉版本之一。
        neutral_rgba: 中性化颜色。

    Yields:
        已应用反事实的环境状态。
    """
    import mujoco

    if variant not in VISUAL_VARIANTS:
        raise ValueError(f"未知视觉版本: {variant}")
    restore_scene_base(env, base)
    try:
        if variant == "swap_positions":
            for index, joint_name in enumerate(
                ("task_red_cube_free_joint", "task_green_cube_free_joint")
            ):
                joint_id = _object_id(env.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                address = int(env.model.jnt_qposadr[joint_id])
                env.data.qpos[address:address + 7] = base.cube_qpos[1 - index]
        elif variant.startswith("neutralize_"):
            color = variant.removeprefix("neutralize_")
            geom_name = f"task_{color}_cube_geom"
            geom_id = _object_id(env.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            env.model.geom_rgba[geom_id] = np.asarray(neutral_rgba, dtype=np.float64)
        mujoco.mj_forward(env.model, env.data)
        if not np.array_equal(env.get_state(), base.robot_state):
            raise RuntimeError("视觉反事实意外改变了机器人状态")
        yield
    finally:
        restore_scene_base(env, base)


def capture_rgb_and_masks(env: CleanTabletopEnv) -> tuple[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]]]:
    """捕获两路RGB和基于geom ID的红绿精确分割mask。

    单路相机可以因视角遮挡而看不到某块积木，但每种颜色必须至少被
    一路真实相机观测到。不可见相机保留全零mask，不伪造ROI。
    """
    import mujoco

    images: dict[str, np.ndarray] = {}
    masks: dict[str, dict[str, np.ndarray]] = {}
    for camera_key, camera_name in CAMERAS.items():
        images[camera_key] = env.capture_camera(camera_name)
        if env._renderer is None:
            raise RuntimeError("MuJoCo renderer尚未初始化")
        try:
            env._renderer.enable_segmentation_rendering()
            env._renderer.update_scene(env.data, camera=camera_name)
            segmentation = env._renderer.render().copy()
        finally:
            env._renderer.disable_segmentation_rendering()
        camera_masks = {}
        for color, geom_name in zip(CUBE_COLORS, TASK_CUBE_GEOM_NAMES, strict=True):
            geom_id = _object_id(env.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            mask = (segmentation[..., 0] == geom_id) & (
                segmentation[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
            )
            camera_masks[color] = mask
        masks[camera_key] = camera_masks
    for color in CUBE_COLORS:
        if not any(masks[camera][color].any() for camera in CAMERAS):
            raise RuntimeError(f"两路真实相机均未观测到{color}积木ROI")
    return images, masks


def mask_to_token_weights(
    mask: np.ndarray,
    grid_size: int = TOKEN_GRID_SIZE,
    allow_empty: bool = False,
) -> np.ndarray:
    """把256像素分割mask聚合为8×8视觉Token权重。

    Args:
        mask: 二维布尔mask。
        grid_size: connector输出空间网格边长。
        allow_empty: 是否把单路相机不可见ROI映射为全零权重。

    Returns:
        总和为1的64维ROI权重。
    """
    binary = np.asarray(mask, dtype=np.bool_)
    if binary.ndim != 2 or binary.shape[0] != binary.shape[1]:
        raise ValueError(f"ROI mask必须是方形二维数组，实际{binary.shape}")
    if binary.shape[0] % grid_size != 0:
        raise ValueError("ROI mask无法映射到视觉Token或内容为空")
    if not binary.any():
        if allow_empty:
            return np.zeros(grid_size * grid_size, dtype=np.float32)
        raise ValueError("ROI mask无法映射到视觉Token或内容为空")
    block = binary.shape[0] // grid_size
    occupancy = binary.reshape(grid_size, block, grid_size, block).mean(axis=(1, 3))
    weights = occupancy.reshape(-1).astype(np.float64)
    if weights.size != IMAGE_TOKEN_COUNT or weights.sum() <= 0.0:
        raise ValueError("ROI没有映射到64个视觉Token")
    return (weights / weights.sum()).astype(np.float32)


def weighted_tokens(tokens: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """按ROI覆盖率聚合视觉或KV Token。"""
    array = np.asarray(tokens)
    weight = np.asarray(weights, dtype=np.float64)
    if array.shape[0] != weight.size or not np.isfinite(array).all():
        raise ValueError("Token数量与ROI权重不一致或包含非有限值")
    return np.tensordot(weight, array, axes=(0, 0)).astype(np.float32)


def dilate_mask(mask: np.ndarray, radius: int = 12) -> np.ndarray:
    """仅用NumPy扩张ROI，为物体边缘与阴影保留容差带。

    Args:
        mask: 二维布尔mask。
        radius: 以像素为单位的方形扩张半径。

    Returns:
        扩张后的布尔mask。
    """
    binary = np.asarray(mask, dtype=np.bool_)
    if binary.ndim != 2 or radius < 0:
        raise ValueError("mask必须为二维数组且radius不能为负")
    if radius == 0:
        return binary.copy()
    padded = np.pad(binary, radius, mode="constant", constant_values=False)
    expanded = np.zeros_like(binary)
    for row_offset in range(2 * radius + 1):
        for column_offset in range(2 * radius + 1):
            expanded |= padded[
                row_offset:row_offset + binary.shape[0],
                column_offset:column_offset + binary.shape[1],
            ]
    return expanded


def audit_visual_intervention(
    original_images: dict[str, np.ndarray],
    original_masks: dict[str, dict[str, np.ndarray]],
    changed_images: dict[str, np.ndarray],
    changed_masks: dict[str, dict[str, np.ndarray]],
    variant: str,
) -> dict[str, Any]:
    """检查像素变化存在且主要落在被干预积木附近。

    Args:
        original_images: 原始两路RGB。
        original_masks: 原始两路红绿mask。
        changed_images: 反事实两路RGB。
        changed_masks: 反事实两路红绿mask。
        variant: 当前视觉版本。

    Returns:
        每路相机的变化像素与ROI外漂移统计。

    Raises:
        RuntimeError: 图像没有响应，或非目标区域出现异常漂移。
    """
    if variant == "original":
        raise ValueError("original不属于视觉干预审计对象")
    affected_colors = CUBE_COLORS if variant == "swap_positions" else (
        variant.removeprefix("neutralize_"),
    )
    audit: dict[str, Any] = {}
    any_camera_changed = False
    for camera in CAMERAS:
        original = np.asarray(original_images[camera])
        changed = np.asarray(changed_images[camera])
        if original.shape != changed.shape:
            raise RuntimeError(f"{camera}图像shape在干预后发生变化")
        changed_pixels = np.any(
            np.abs(original.astype(np.int16) - changed.astype(np.int16)) > PIXEL_CHANGE_THRESHOLD,
            axis=-1,
        )
        allowed = np.zeros(changed_pixels.shape, dtype=np.bool_)
        for color in affected_colors:
            allowed |= original_masks[camera][color]
            allowed |= changed_masks[camera][color]
        allowed = dilate_mask(allowed)
        affected_visible = bool(allowed.any())
        if not changed_pixels.any() and affected_visible:
            raise RuntimeError(f"{variant}未引起可见{camera} RGB变化")
        any_camera_changed |= bool(changed_pixels.any())
        outside_changed = changed_pixels & ~allowed
        outside_rate = float(outside_changed.sum() / max(int(changed_pixels.sum()), 1))
        if outside_rate > OUTSIDE_ROI_DRIFT_LIMIT:
            raise RuntimeError(
                f"{variant}导致{camera}非目标区域异常漂移: {outside_rate:.4%}"
            )
        audit[camera] = {
            "changed_pixel_count": int(changed_pixels.sum()),
            "outside_roi_changed_pixel_count": int(outside_changed.sum()),
            "outside_roi_changed_fraction": outside_rate,
            "affected_object_visible": affected_visible,
        }
    if not any_camera_changed:
        raise RuntimeError(f"{variant}未引起任何真实相机RGB变化")
    return audit


def extract_features(
    policy: Any,
    processed: dict[str, Any],
    roi_weights: dict[str, dict[str, np.ndarray]],
    color_token_index: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """提取视觉Token以及真实prefix路径中的跨模态特征。"""
    import torch
    from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    images, img_masks = policy.prepare_images(processed)
    state = policy.prepare_state(processed)
    lang_tokens = processed[OBS_LANGUAGE_TOKENS]
    lang_masks = processed[OBS_LANGUAGE_ATTENTION_MASK].bool()
    if len(images) < 2 or int(policy.config.prefix_length) != 0:
        raise ValueError("诊断要求至少两路图像且prefix_length=0")
    model = policy.model
    visual: dict[str, np.ndarray] = {}
    connected_tokens = []
    for camera_index, camera_key in enumerate(CAMERAS):
        token_tensor = model.vlm_with_expert.embed_image(images[camera_index])
        tokens = token_tensor[0].detach().float().cpu().numpy()
        if tokens.shape[0] != IMAGE_TOKEN_COUNT:
            raise ValueError(f"{camera_key}视觉Token数量不是64: {tokens.shape}")
        connected_tokens.append(tokens)
        visual[f"{camera_key}_tokens"] = tokens
        for color in CUBE_COLORS:
            visual[f"{camera_key}_{color}_roi"] = weighted_tokens(
                tokens, roi_weights[camera_key][color]
            )

    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
        images, img_masks, lang_tokens, lang_masks, state=state
    )
    language_length = int(lang_tokens.shape[1])
    state_length = 1
    language_start = int(prefix_embs.shape[1]) - language_length - state_length
    if language_start != IMAGE_TOKEN_COUNT * len(images):
        raise ValueError(
            f"prefix图像Token布局不符合预期: language_start={language_start}, images={len(images)}"
        )
    active_indices = torch.nonzero(lang_masks[0], as_tuple=False).flatten()
    if color_token_index not in active_indices.tolist():
        raise ValueError("颜色Token落在padding区")
    color_prefix_index = language_start + color_token_index
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
    conditioning = {
        "color_token_id": lang_tokens[0, color_token_index].detach().cpu().numpy(),
        "color_final_hidden": outputs[0][0, color_prefix_index].detach().float().cpu().numpy(),
    }
    for layer_index in sorted(cache):
        layer = cache[layer_index]
        for kind, tensor_name in (("key", "key_states"), ("value", "value_states")):
            tensor = layer[tensor_name][0]
            conditioning[f"layer_{layer_index:02d}_color_{kind}"] = (
                tensor[color_prefix_index].detach().float().cpu().numpy()
            )
            for camera_index, camera_key in enumerate(CAMERAS):
                start = camera_index * IMAGE_TOKEN_COUNT
                end = start + IMAGE_TOKEN_COUNT
                camera_tokens = tensor[start:end].detach().float().cpu().numpy()
                for color in CUBE_COLORS:
                    conditioning[f"layer_{layer_index:02d}_{camera_key}_{color}_{kind}"] = (
                        weighted_tokens(camera_tokens, roi_weights[camera_key][color])
                    )
    return visual, conditioning


def causal_selectivity(target_distance: float, distractor_distance: float) -> float:
    """计算目标干预相对干扰物干预的归一化因果选择指数。"""
    target = float(target_distance)
    distractor = float(distractor_distance)
    if not np.isfinite([target, distractor]).all() or target < 0.0 or distractor < 0.0:
        raise ValueError("CSI距离必须是有限非负数")
    denominator = target + distractor
    return 0.0 if denominator <= 1e-12 else (target - distractor) / denominator


def bootstrap_median_ci(values: Sequence[float], seed: int = 20260) -> tuple[float, float]:
    """按scene重采样计算中位数95%区间。"""
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("Bootstrap输入必须非空且有限")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, array.size, size=(BOOTSTRAP_SAMPLES, array.size))
    estimates = np.median(array[indices], axis=1)
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def consistency_label(scene_values: Sequence[float]) -> str:
    """根据六个scene方向与Bootstrap区间生成诊断标签。"""
    values = np.asarray(scene_values, dtype=np.float64)
    if values.size != 6:
        return "insufficient_scenes"
    positive = int((values > 0.0).sum())
    lower, _ = bootstrap_median_ci(values)
    if positive >= 5 and lower > 0.0:
        return "consistent"
    if positive <= 2:
        return "opposite_or_insensitive"
    return "mixed"


def identify_color_token_index(processed_red: dict[str, Any], processed_green: dict[str, Any]) -> int:
    """从真实预处理结果动态定位唯一颜色Token索引。"""
    from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

    red_tokens = processed_red[OBS_LANGUAGE_TOKENS][0].detach().cpu().numpy()
    green_tokens = processed_green[OBS_LANGUAGE_TOKENS][0].detach().cpu().numpy()
    red_mask = processed_red[OBS_LANGUAGE_ATTENTION_MASK][0].detach().cpu().numpy().astype(bool)
    green_mask = processed_green[OBS_LANGUAGE_ATTENTION_MASK][0].detach().cpu().numpy().astype(bool)
    if not np.array_equal(red_mask, green_mask):
        raise ValueError("红绿canonical指令attention mask不一致")
    differences = np.flatnonzero((red_tokens != green_tokens) & red_mask)
    if differences.size != 1:
        raise ValueError(f"红绿指令必须只有一个有效颜色Token差异，实际={differences.tolist()}")
    return int(differences[0])


def fk_trajectory(env: CleanTabletopEnv, chunk: np.ndarray) -> np.ndarray:
    """把绝对关节目标转换为attachment-site命令轨迹，不推进动力学。"""
    import mujoco

    addresses = []
    for joint_name in ARM_JOINT_NAMES:
        joint_id = _object_id(env.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        addresses.append(int(env.model.jnt_qposadr[joint_id]))
    site_id = _object_id(env.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    original_qpos = env.data.qpos.copy()
    trajectory = []
    try:
        for action in np.asarray(chunk, dtype=np.float64):
            env.data.qpos[addresses] = action[:6]
            mujoco.mj_forward(env.model, env.data)
            trajectory.append(env.data.site_xpos[site_id].copy())
    finally:
        env.data.qpos[:] = original_qpos
        mujoco.mj_forward(env.model, env.data)
    return np.asarray(trajectory, dtype=np.float64)


def spatial_metrics(
    original_trajectory: np.ndarray,
    swapped_trajectory: np.ndarray,
    original_target: np.ndarray,
    swapped_target: np.ndarray,
    original_distractor: np.ndarray,
    swapped_distractor: np.ndarray,
    horizon: int,
) -> dict[str, float]:
    """计算目标位置交换后的末端跟随方向和距离优势。"""
    original = np.asarray(original_trajectory, dtype=np.float64)[:horizon]
    swapped = np.asarray(swapped_trajectory, dtype=np.float64)[:horizon]
    target_displacement = np.asarray(swapped_target) - np.asarray(original_target)
    action_displacement = swapped[-1] - original[-1]
    target_norm = float(np.linalg.norm(target_displacement))
    action_norm = float(np.linalg.norm(action_displacement))
    cosine = float(
        np.dot(action_displacement, target_displacement) / (action_norm * target_norm + 1e-12)
    )
    follow_gain = float(np.dot(action_displacement, target_displacement) / (target_norm**2 + 1e-12))
    original_margin = float(
        np.min(np.linalg.norm(original - original_distractor, axis=1))
        - np.min(np.linalg.norm(original - original_target, axis=1))
    )
    swapped_margin = float(
        np.min(np.linalg.norm(swapped - swapped_distractor, axis=1))
        - np.min(np.linalg.norm(swapped - swapped_target, axis=1))
    )
    return {
        "horizon": horizon,
        "endpoint_migration_m": action_norm,
        "target_displacement_m": target_norm,
        "position_follow_cosine": cosine,
        "follow_gain": follow_gain,
        "original_correct_distance_margin_m": original_margin,
        "swapped_correct_distance_margin_m": swapped_margin,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """写出UTF-8 BOM CSV。"""
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        if not rows:
            return
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def project_source_sha256() -> str:
    """计算影响视觉反事实行为的源码身份。"""
    project_root = Path(__file__).resolve().parents[1]
    paths = (
        Path(__file__).resolve(),
        project_root / "evaluate" / "diagnose_conditioning.py",
        project_root / "evaluate" / "rollout.py",
        project_root / "evaluate" / "common.py",
        project_root / "sim" / "environment.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(project_root)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def git_identity() -> dict[str, Any]:
    """读取当前Git提交和工作区状态。"""
    root = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, encoding="utf-8"
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=root, text=True, encoding="utf-8"
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
    noise_hash: str,
    amp_enabled: bool,
) -> dict[str, Any]:
    """构造checkpoint、源码、配置、环境和噪声清单。"""
    import mujoco
    import torch

    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint / "model.safetensors"),
        "source_sha256": project_source_sha256(),
        "git": git_identity(),
        "config": config,
        "settings": settings,
        "fixed_noise": {"shape": [1, 50, 32], "policy_seed": 20260, "sha256": noise_hash},
        "repeat_tolerance": {"atol": REPEAT_ATOL, "rtol": REPEAT_RTOL},
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "device": device,
            "amp_enabled": amp_enabled,
            "lerobot": importlib.metadata.version("lerobot"),
            "mujoco": mujoco.__version__,
        },
        "vision_token_mapping": {
            "render_size": [256, 256],
            "preprocessor_size": [512, 512],
            "connector_grid": [8, 8],
            "tokens_per_camera": 64,
        },
        "mujoco_stepped": False,
        "closed_loop_success_measured": False,
    }


def _feature_distance(group: dict[str, np.ndarray], left: str, right: str, names: Sequence[str]) -> float:
    """连接指定特征后计算L2距离。"""
    lhs = np.concatenate([np.asarray(group[f"{left}__{name}"]).reshape(-1) for name in names])
    rhs = np.concatenate([np.asarray(group[f"{right}__{name}"]).reshape(-1) for name in names])
    return float(np.linalg.norm(lhs.astype(np.float64) - rhs.astype(np.float64)))


def build_visual_rows(
    visual: dict[str, dict[str, np.ndarray]],
    settings: dict[str, Any],
) -> list[dict[str, Any]]:
    """比较原始与三种视觉反事实的视觉Token和ROI特征。"""
    rows = []
    for scene_seed in settings["scene_seeds"]:
        original = visual[f"scene_{scene_seed}__original"]
        for variant in VISUAL_VARIANTS[1:]:
            changed = visual[f"scene_{scene_seed}__{variant}"]
            for camera in CAMERAS:
                for scope in ("tokens", "red_roi", "green_roi"):
                    name = f"{camera}_{scope}"
                    rows.append(
                        {
                            "scene_seed": scene_seed,
                            "variant": variant,
                            "camera": camera,
                            "scope": scope,
                            **flatten_distance(original[name], changed[name]),
                        }
                    )
    return rows


def build_causal_rows(
    conditions: Sequence[VisualCondition],
    conditioning: dict[str, dict[str, np.ndarray]],
    actions: dict[str, dict[str, np.ndarray]],
    horizons: Sequence[int],
) -> list[dict[str, Any]]:
    """计算VLM/KV与三阶段动作的目标—干扰物CSI。"""
    indexed = {
        (item.scene_seed, item.source_color, item.visual_variant): item.key for item in conditions
    }
    rows: list[dict[str, Any]] = []
    for scene_seed in sorted({item.scene_seed for item in conditions}):
        for source_color in CUBE_COLORS:
            distractor_color = "green" if source_color == "red" else "red"
            original_key = indexed[(scene_seed, source_color, "original")]
            target_key = indexed[(scene_seed, source_color, f"neutralize_{source_color}")]
            distractor_key = indexed[(scene_seed, source_color, f"neutralize_{distractor_color}")]
            feature_names = sorted(conditioning[original_key])
            signal_groups = {
                "vlm_color_final_hidden": ["color_final_hidden"],
                "vlm_color_kv_all_layers": [
                    name for name in feature_names if "_color_key" in name or "_color_value" in name
                ],
                "vlm_target_roi_kv_all_layers": [
                    name
                    for name in feature_names
                    if f"_{source_color}_key" in name or f"_{source_color}_value" in name
                ],
            }
            layer_prefixes = sorted(
                {name.split("_color_")[0] for name in feature_names if "_color_key" in name}
            )
            for layer_prefix in layer_prefixes:
                signal_groups[f"{layer_prefix}_color_kv"] = [
                    name for name in feature_names if name.startswith(f"{layer_prefix}_color_")
                ]
                signal_groups[f"{layer_prefix}_target_roi_kv"] = [
                    name
                    for name in feature_names
                    if name.startswith(f"{layer_prefix}_")
                    and (f"_{source_color}_key" in name or f"_{source_color}_value" in name)
                ]
            flattened = {
                f"{key}__{name}": value
                for key in (original_key, target_key, distractor_key)
                for name, value in conditioning[key].items()
            }
            for signal, names in signal_groups.items():
                target_distance = _feature_distance(flattened, original_key, target_key, names)
                distractor_distance = _feature_distance(flattened, original_key, distractor_key, names)
                rows.append(
                    {
                        "scene_seed": scene_seed,
                        "source_color": source_color,
                        "signal": signal,
                        "stage": "feature",
                        "horizon": "",
                        "target_distance": target_distance,
                        "distractor_distance": distractor_distance,
                        "csi": causal_selectivity(target_distance, distractor_distance),
                        "target_per_dimension_mae": "",
                        "distractor_per_dimension_mae": "",
                    }
                )
            for stage in ("normalized", "physical", "clipped"):
                for horizon in horizons:
                    target_metrics = action_distance(
                        actions[original_key][stage], actions[target_key][stage], horizon
                    )
                    distractor_metrics = action_distance(
                        actions[original_key][stage], actions[distractor_key][stage], horizon
                    )
                    rows.append(
                        {
                            "scene_seed": scene_seed,
                            "source_color": source_color,
                            "signal": "action",
                            "stage": stage,
                            "horizon": horizon,
                            "target_distance": target_metrics["l2"],
                            "distractor_distance": distractor_metrics["l2"],
                            "csi": causal_selectivity(
                                target_metrics["l2"], distractor_metrics["l2"]
                            ),
                            "target_per_dimension_mae": json.dumps(
                                target_metrics["per_dimension_mae"], ensure_ascii=False
                            ),
                            "distractor_per_dimension_mae": json.dumps(
                                distractor_metrics["per_dimension_mae"], ensure_ascii=False
                            ),
                        }
                    )
    return rows


def build_spatial_rows(
    env: CleanTabletopEnv,
    conditions: Sequence[VisualCondition],
    actions: dict[str, dict[str, np.ndarray]],
    variant_layouts: dict[str, dict[str, list[float]]],
    horizons: Sequence[int],
) -> list[dict[str, Any]]:
    """计算original到位置交换条件的FK跟随指标。"""
    indexed = {
        (item.scene_seed, item.source_color, item.visual_variant): item.key for item in conditions
    }
    rows = []
    for scene_seed in sorted({item.scene_seed for item in conditions}):
        for source_color in CUBE_COLORS:
            distractor_color = "green" if source_color == "red" else "red"
            original_key = indexed[(scene_seed, source_color, "original")]
            swapped_key = indexed[(scene_seed, source_color, "swap_positions")]
            original_trajectory = fk_trajectory(env, actions[original_key]["clipped"])
            swapped_trajectory = fk_trajectory(env, actions[swapped_key]["clipped"])
            original_layout = variant_layouts[f"scene_{scene_seed}__original"]
            swapped_layout = variant_layouts[f"scene_{scene_seed}__swap_positions"]
            target_name = f"task_{source_color}_cube"
            distractor_name = f"task_{distractor_color}_cube"
            for horizon in horizons:
                rows.append(
                    {
                        "scene_seed": scene_seed,
                        "source_color": source_color,
                        **spatial_metrics(
                            original_trajectory,
                            swapped_trajectory,
                            np.asarray(original_layout[target_name]),
                            np.asarray(swapped_layout[target_name]),
                            np.asarray(original_layout[distractor_name]),
                            np.asarray(swapped_layout[distractor_name]),
                            horizon,
                        ),
                    }
                )
    return rows


def aggregate_scene_metric(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    """先按scene平均红绿，再生成方向一致性统计。"""
    scenes = sorted({int(row["scene_seed"]) for row in rows})
    values = [
        float(np.mean([float(row[field]) for row in rows if int(row["scene_seed"]) == scene]))
        for scene in scenes
    ]
    ci = bootstrap_median_ci(values)
    return {
        "scene_values": values,
        "median": float(np.median(values)),
        "positive_scenes": int(np.sum(np.asarray(values) > 0.0)),
        "scene_count": len(values),
        "bootstrap_ci95": list(ci),
        "label": consistency_label(values),
    }


def diagnose_causal_stage(
    visual_sensitive: bool,
    vlm_label: str,
    action_label: str,
    position_label: str,
) -> str:
    """根据四级证据返回最早失败环节。

    Args:
        visual_sensitive: 视觉Token是否响应像素反事实。
        vlm_label: 跨模态VLM-CSI标签。
        action_label: 动作因果CSI标签。
        position_label: FK位置跟随标签。

    Returns:
        分层故障定位标签。
    """
    if not visual_sensitive:
        return "vision_encoder_or_connector_insensitive"
    if vlm_label == "opposite_or_insensitive":
        return "cross_modal_grounding_insufficient"
    if vlm_label != "consistent":
        return "cross_modal_grounding_mixed"
    if action_label == "opposite_or_insensitive":
        return "action_expert_ignores_visual_grounding"
    if action_label != "consistent":
        return "action_expert_visual_use_mixed"
    if position_label == "opposite_or_insensitive":
        return "spatial_action_mapping_insufficient"
    if position_label != "consistent":
        return "spatial_action_mapping_mixed"
    return "causal_chain_consistent"


def build_summary(
    settings: dict[str, Any],
    records: list[dict[str, Any]],
    visual_rows: list[dict[str, Any]],
    causal_rows: list[dict[str, Any]],
    spatial_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """聚合有效性、CSI、位置跟随和故障定位。"""
    repeats = [record for record in records if record["record_type"] == "repeat_control"]
    nondeterministic = [record["inference_key"] for record in repeats if not record["repeat_deterministic"]]
    visual_distances = [
        float(row["relative_l2"])
        for row in visual_rows
        if row["scope"] == "tokens" and row["variant"].startswith("neutralize_")
    ]
    vlm_rows = [row for row in causal_rows if row["signal"] == "vlm_color_final_hidden"]
    action_rows = [
        row
        for row in causal_rows
        if row["signal"] == "action" and row["stage"] == "physical" and row["horizon"] == 50
    ]
    position_rows = [row for row in spatial_rows if row["horizon"] == 50]
    vlm = aggregate_scene_metric(vlm_rows, "csi")
    action = aggregate_scene_metric(action_rows, "csi")
    position = aggregate_scene_metric(position_rows, "position_follow_cosine")
    diagnosis = (
        "insufficient_scenes_for_diagnosis"
        if len(settings["scene_seeds"]) != 6
        else diagnose_causal_stage(
            float(np.median(visual_distances)) > 1e-8,
            vlm["label"],
            action["label"],
            position["label"],
        )
    )
    return {
        "status": "invalid" if nondeterministic else "complete",
        "unique_condition_count": len([r for r in records if r["record_type"] == "condition"]),
        "repeat_control_count": len(repeats),
        "action_chunk_count": len(records),
        "single_noise_hash": len({record["noise_sha256"] for record in records}) == 1,
        "nondeterministic_repeat_count": len(nondeterministic),
        "nondeterministic_inference_keys": nondeterministic,
        "median_visual_token_relative_l2_neutralization": float(np.median(visual_distances)),
        "vlm_color_csi": vlm,
        "physical_action_csi_full50": action,
        "position_following_full50": position,
        "diagnosis": diagnosis,
        "boundary": "FK命令轨迹和CSI用于因果定位，不等同于MuJoCo闭环成功率。",
        "settings": settings,
    }


def write_report(path: Path, summary: dict[str, Any]) -> None:
    """写出中文视觉反事实诊断报告。"""
    vlm = summary["vlm_color_csi"]
    action = summary["physical_action_csi_full50"]
    position = summary["position_following_full50"]
    lines = [
        "# SmolVLA视觉反事实因果诊断",
        "",
        f"- 状态：`{summary['status']}`",
        f"- 唯一条件：{summary['unique_condition_count']}",
        f"- 重复控制：{summary['repeat_control_count']}",
        f"- Action Chunk：{summary['action_chunk_count']}",
        f"- 非确定性重复：{summary['nondeterministic_repeat_count']}",
        f"- 分层诊断：`{summary['diagnosis']}`",
        "",
        "## 关键因果指标",
        "",
        f"- 视觉Token中性化相对L2中位数：`{summary['median_visual_token_relative_l2_neutralization']:.8g}`",
        f"- VLM颜色CSI：中位数`{vlm['median']:.6f}`，正向scene `{vlm['positive_scenes']}/{vlm['scene_count']}`，标签`{vlm['label']}`",
        f"- 物理动作CSI（50步）：中位数`{action['median']:.6f}`，正向scene `{action['positive_scenes']}/{action['scene_count']}`，标签`{action['label']}`",
        f"- 位置跟随余弦（50步）：中位数`{position['median']:.6f}`，正向scene `{position['positive_scenes']}/{position['scene_count']}`，标签`{position['label']}`",
        "",
        "## 解读边界",
        "",
        summary["boundary"],
        "视觉Token有变化只能证明编码器对像素干预敏感；只有目标中性化比干扰物中性化造成更大VLM和动作变化，才支持颜色空间绑定。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_diagnostic(
    checkpoint: Path,
    config: dict[str, Any],
    settings: dict[str, Any],
    output_dir: Path,
    device: str,
) -> dict[str, Any]:
    """执行完整视觉反事实因果诊断。"""
    import imageio.v2 as imageio
    import torch

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("请求cuda诊断，但当前环境没有可用CUDA")
    conditions = build_conditions(settings)
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
        raise ValueError(f"固定噪声shape不符合(1,50,32): {tuple(fixed_noise.shape)}")
    noise_cpu = fixed_noise.detach().cpu().numpy()
    np.save(output_dir / "fixed_noise.npy", noise_cpu)
    noise_hash = array_sha256(noise_cpu)
    write_json(
        output_dir / "run_manifest.json",
        build_manifest(
            checkpoint,
            config,
            settings,
            device,
            noise_hash,
            bool(policy.config.use_amp),
        ),
    )

    inputs_dir = output_dir / "counterfactual_inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    visual_features: dict[str, dict[str, np.ndarray]] = {}
    conditioning_features: dict[str, dict[str, np.ndarray]] = {}
    actions: dict[str, dict[str, np.ndarray]] = {}
    variant_layouts: dict[str, dict[str, list[float]]] = {}
    variant_inputs: dict[tuple[int, str], tuple[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]], np.ndarray]] = {}
    intervention_audits: dict[str, Any] = {}

    with CleanTabletopEnv() as env:
        arm_ctrlrange = env.model.actuator_ctrlrange[:6].copy()
        for scene_seed in settings["scene_seeds"]:
            env.reset(scene_seed)
            base = capture_scene_base(env)
            base_hash = array_sha256(base.robot_state)
            for variant in settings["visual_variants"]:
                with visual_variant_context(env, base, variant, settings["neutral_rgba"]):
                    images, masks = capture_rgb_and_masks(env)
                    state = env.get_state()
                    if array_sha256(state) != base_hash:
                        raise RuntimeError("视觉版本之间机器人状态哈希不一致")
                    key = f"scene_{scene_seed}__{variant}"
                    layout = env.task_layout()
                    variant_layouts[key] = {
                        name: np.asarray(value["position"], dtype=float).tolist()
                        for name, value in layout.items()
                    }
                    variant_inputs[(scene_seed, variant)] = (images, masks, state.copy())
                    for camera in CAMERAS:
                        imageio.imwrite(inputs_dir / f"{key}_{camera}.png", images[camera])
                        for color in CUBE_COLORS:
                            imageio.imwrite(
                                inputs_dir / f"{key}_{camera}_{color}_mask.png",
                                masks[camera][color].astype(np.uint8) * 255,
                            )
                    write_json(
                        inputs_dir / f"{key}.json",
                        {
                            "scene_seed": scene_seed,
                            "variant": variant,
                            "robot_state": state.astype(float).tolist(),
                            "robot_state_sha256": array_sha256(state),
                            "image_sha256": {camera: array_sha256(images[camera]) for camera in CAMERAS},
                            "mask_pixels": {
                                camera: {color: int(masks[camera][color].sum()) for color in CUBE_COLORS}
                                for camera in CAMERAS
                            },
                            "layout_positions": variant_layouts[key],
                        },
                    )
            restore_scene_base(env, base)

            original_images, original_masks, _ = variant_inputs[(scene_seed, "original")]
            original_layout = variant_layouts[f"scene_{scene_seed}__original"]
            for variant in settings["visual_variants"][1:]:
                changed_images, changed_masks, _ = variant_inputs[(scene_seed, variant)]
                changed_layout = variant_layouts[f"scene_{scene_seed}__{variant}"]
                if variant == "swap_positions":
                    if not np.allclose(
                        changed_layout["task_red_cube"], original_layout["task_green_cube"]
                    ) or not np.allclose(
                        changed_layout["task_green_cube"], original_layout["task_red_cube"]
                    ):
                        raise RuntimeError("swap_positions没有精确交换红绿积木位置")
                else:
                    if any(
                        not np.allclose(changed_layout[name], original_layout[name])
                        for name in TASK_CUBE_BODY_NAMES
                    ):
                        raise RuntimeError(f"{variant}意外改变了积木位置")
                audit_key = f"scene_{scene_seed}__{variant}"
                intervention_audits[audit_key] = audit_visual_intervention(
                    original_images,
                    original_masks,
                    changed_images,
                    changed_masks,
                    variant,
                )
            write_json(
                inputs_dir / f"scene_{scene_seed}__intervention_audit.json",
                {
                    key: value
                    for key, value in intervention_audits.items()
                    if key.startswith(f"scene_{scene_seed}__")
                },
            )

        first_scene = settings["scene_seeds"][0]
        images, _, state = variant_inputs[(first_scene, "original")]
        red_processed = preprocessor(
            make_policy_observation(images, state, build_prompt("red_on_blue", "canonical"))
        )
        green_processed = preprocessor(
            make_policy_observation(images, state, build_prompt("green_on_blue", "canonical"))
        )
        color_token_index = identify_color_token_index(red_processed, green_processed)

        with (output_dir / "condition_records.jsonl").open("w", encoding="utf-8", newline="\n") as jsonl:
            for condition in conditions:
                images, masks, state = variant_inputs[(condition.scene_seed, condition.visual_variant)]
                prompt = build_prompt(condition.task_id, condition.prompt_type)
                set_policy_seed(condition.policy_seed)
                processed = preprocessor(make_policy_observation(images, state, prompt))
                weights = {
                    camera: {
                        color: mask_to_token_weights(
                            masks[camera][color], allow_empty=True
                        )
                        for color in CUBE_COLORS
                    }
                    for camera in CAMERAS
                }
                autocast = (
                    torch.autocast(device_type="cuda")
                    if device.startswith("cuda") and bool(policy.config.use_amp)
                    else nullcontext()
                )
                with torch.inference_mode(), autocast:
                    visual, conditioning = extract_features(
                        policy, processed, weights, color_token_index
                    )
                visual_key = f"scene_{condition.scene_seed}__{condition.visual_variant}"
                if visual_key not in visual_features:
                    visual_features[visual_key] = {
                        **visual,
                        **{
                            f"{camera}_{color}_weights": weights[camera][color]
                            for camera in CAMERAS
                            for color in CUBE_COLORS
                        },
                    }
                conditioning_features[condition.key] = conditioning
                inference = infer_action_chunk(
                    policy, postprocessor, processed, fixed_noise, arm_ctrlrange, device
                )
                actions[condition.key] = {
                    "normalized": inference[0],
                    "physical": inference[1],
                    "clipped": inference[2],
                    "clipped_mask": inference[3],
                }
                record = {
                    "inference_key": condition.key,
                    "condition_key": condition.key,
                    "record_type": "condition",
                    "repeat_index": 0,
                    "scene_seed": condition.scene_seed,
                    "source_color": condition.source_color,
                    "visual_variant": condition.visual_variant,
                    "task_id": condition.task_id,
                    "prompt": prompt,
                    "policy_seed": condition.policy_seed,
                    "noise_sha256": noise_hash,
                    "robot_state_sha256": array_sha256(state),
                    "repeat_deterministic": True,
                    "repeat_normalized_l2": 0.0,
                }
                records.append(record)
                jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")
                jsonl.flush()

                if condition.visual_variant == "original":
                    repeated = infer_action_chunk(
                        policy, postprocessor, processed, fixed_noise, arm_ctrlrange, device
                    )
                    repeat_key = f"{condition.key}__repeat_1"
                    actions[repeat_key] = {
                        "normalized": repeated[0],
                        "physical": repeated[1],
                        "clipped": repeated[2],
                        "clipped_mask": repeated[3],
                    }
                    distance = flatten_distance(inference[0], repeated[0])
                    repeat_record = {
                        **record,
                        "inference_key": repeat_key,
                        "record_type": "repeat_control",
                        "repeat_index": 1,
                        "repeat_deterministic": repeat_is_deterministic(inference[0], repeated[0]),
                        "repeat_normalized_l2": distance["l2"],
                    }
                    records.append(repeat_record)
                    jsonl.write(json.dumps(repeat_record, ensure_ascii=False) + "\n")
                    jsonl.flush()

        visual_rows = build_visual_rows(visual_features, settings)
        causal_rows = build_causal_rows(
            conditions, conditioning_features, actions, settings["analysis_horizons"]
        )
        spatial_rows = build_spatial_rows(
            env, conditions, actions, variant_layouts, settings["analysis_horizons"]
        )

    save_npz(output_dir / "visual_features.npz", visual_features)
    save_npz(output_dir / "conditioning_features.npz", conditioning_features)
    save_npz(output_dir / "action_chunks.npz", actions)
    write_csv(output_dir / "visual_feature_comparison.csv", visual_rows)
    write_csv(output_dir / "causal_comparison.csv", causal_rows)
    write_csv(output_dir / "spatial_following.csv", spatial_rows)
    summary = build_summary(settings, records, visual_rows, causal_rows, spatial_rows)
    summary["intervention_audit_count"] = len(intervention_audits)
    summary["max_outside_roi_changed_fraction"] = max(
        camera_audit["outside_roi_changed_fraction"]
        for audit in intervention_audits.values()
        for camera_audit in audit.values()
    )
    write_json(output_dir / "summary.json", summary)
    write_report(output_dir / "report.md", summary)
    action_rows = [
        row
        for row in causal_rows
        if row["signal"] == "action" and row["stage"] == "physical" and row["horizon"] == 50
    ]
    action_scene_values = aggregate_scene_metric(action_rows, "csi")["scene_values"]
    position_rows = [row for row in spatial_rows if row["horizon"] == 50]
    position_scene_values = aggregate_scene_metric(position_rows, "position_follow_cosine")["scene_values"]
    write_pillow_line_plot(
        output_dir / "causal_selectivity.png",
        np.asarray(action_scene_values),
        "Physical action causal selectivity by scene",
        marker_index=None,
    )
    write_pillow_line_plot(
        output_dir / "position_following.png",
        np.asarray(position_scene_values),
        "Position-follow cosine by scene",
        marker_index=None,
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    """解析参数并执行视觉反事实诊断。"""
    args = build_parser().parse_args(argv)
    checkpoint = find_pretrained_model(args.checkpoint)
    config = load_yaml_config(resolve_path(args.config))
    settings = parse_config(config)
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
