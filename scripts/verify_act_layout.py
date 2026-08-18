"""验证迁移场景与 ACT 源场景的空间布局完全一致。"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Sequence
from pathlib import Path

import mujoco
import numpy as np
from numpy.typing import NDArray

from sim.environment import ACT_INITIAL_ARM_QPOS_DEG, ARM_JOINT_NAMES, CleanTabletopEnv


def _build_source_model(source_root: Path) -> mujoco.MjModel:
    """在内存中兼容修复 ACT 源场景并编译模型。

    该函数只读取源项目的 ``demo_scene.xml``。杯盘 include、原文件中的
    ``#`` 文本说明和 ``meshdir`` 修复都发生在内存中，磁盘上的 ACT 项目
    不会被修改。

    Args:
        source_root: ACT 项目的 ``mode`` 目录。

    Returns:
        删除杯盘后的 ACT 源场景模型。

    Raises:
        FileNotFoundError: 源场景目录或必需文件不存在时抛出。
    """
    scene_path = source_root / "demo_scene.xml"
    robot_path = source_root / "ur10e_with_2f85_d435i.xml"
    if not scene_path.is_file() or not robot_path.is_file():
        raise FileNotFoundError(f"ACT 场景目录不完整: {source_root}")

    scene_xml = scene_path.read_text(encoding="utf-8")
    scene_xml = re.sub(r"^.*(?:plate_11|mug_5).*$\n?", "", scene_xml, flags=re.MULTILINE)
    # demo_scene.xml 使用 Python 风格的 # 说明文字。MuJoCo通常会忽略这些
    # 文本，但验证器主动清理它们，使内存参考模型始终是规范XML。
    scene_xml = re.sub(r"[ \t]*#.*", "", scene_xml)
    robot_xml = robot_path.read_text(encoding="utf-8").replace(
        '<compiler angle="radian" meshdir="assets" autolimits="true"/>',
        '<compiler angle="radian" autolimits="true"/>',
    )

    required_files = [
        source_root / "tabletop" / "object" / "object_table.xml",
        source_root / "tabletop" / "mesh" / "light_wood_v3.png",
    ]
    for directory in (
        source_root / "universal_robots_ur10e" / "assets",
        source_root / "robotiq_2f85" / "assets",
        source_root / "realsense_d435i" / "assets",
    ):
        required_files.extend(sorted(path for path in directory.iterdir() if path.is_file()))

    assets = {
        path.relative_to(source_root).as_posix(): path.read_bytes()
        for path in required_files
    }
    assets["ur10e_with_2f85_d435i.xml"] = robot_xml.encode("utf-8")
    return mujoco.MjModel.from_xml_string(scene_xml, assets=assets)


def _reset_model(model: mujoco.MjModel) -> mujoco.MjData:
    """设置 ACT 初始关节角并计算模型世界位姿。

    Args:
        model: 需要初始化的源模型或迁移模型。

    Returns:
        已执行 ``mj_forward`` 的 MuJoCo 数据对象。
    """
    data = mujoco.MjData(model)
    initial_qpos = np.deg2rad(ACT_INITIAL_ARM_QPOS_DEG)
    for joint_name, joint_value in zip(ARM_JOINT_NAMES, initial_qpos, strict=True):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        data.qpos[int(model.jnt_qposadr[joint_id])] = joint_value
    data.ctrl[: len(initial_qpos)] = initial_qpos
    mujoco.mj_forward(model, data)
    return data


def _pose_values(
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> dict[str, NDArray[np.float64]]:
    """提取机械臂、桌面和相机的世界坐标量。

    Args:
        model: MuJoCo 模型。
        data: 已计算正向运动学的数据。

    Returns:
        名称到浮点数组副本的映射。
    """
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    table_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "front_object_table")
    table_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "front_object_table")
    values = {
        "base_position": data.xpos[base_id].copy(),
        "base_rotation": data.xmat[base_id].copy(),
        "table_position": data.xpos[table_id].copy(),
        "table_rotation": data.xmat[table_id].copy(),
        "table_geom_position": data.geom_xpos[table_geom_id].copy(),
        "table_geom_size": model.geom_size[table_geom_id].copy(),
    }
    for camera_name in ("agentview", "topview", "sideview", "d435i_rgb"):
        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        values[f"camera.{camera_name}.position"] = data.cam_xpos[camera_id].copy()
        values[f"camera.{camera_name}.rotation"] = data.cam_xmat[camera_id].copy()
        values[f"camera.{camera_name}.fovy"] = np.array([model.cam_fovy[camera_id]])
    return values


def verify_layout(source_root: Path, tolerance: float = 1e-9) -> dict[str, object]:
    """比较源场景和迁移场景的编译后世界位姿。

    Args:
        source_root: ACT 项目的 ``mode`` 目录。
        tolerance: 平移、旋转和视场角允许的最大绝对误差。

    Returns:
        包含逐项误差和全局最大误差的可序列化报告。

    Raises:
        AssertionError: 任一误差超过容差时抛出。
    """
    source_model = _build_source_model(source_root.resolve())
    source_data = _reset_model(source_model)
    source_values = _pose_values(source_model, source_data)

    with CleanTabletopEnv() as target_env:
        target_values = _pose_values(target_env.model, target_env.data)

    errors = {
        name: float(np.max(np.abs(source_values[name] - target_values[name])))
        for name in source_values
    }
    max_error = max(errors.values(), default=0.0)
    if max_error > tolerance:
        raise AssertionError(f"ACT 空间布局不一致: max_error={max_error}, errors={errors}")
    return {"tolerance": tolerance, "max_error": max_error, "errors": errors}


def build_parser() -> argparse.ArgumentParser:
    """创建空间布局验证脚本的参数解析器。

    Returns:
        已配置源目录和容差参数的解析器。
    """
    parser = argparse.ArgumentParser(description="比较 ACT 源场景和 SmolVLA 迁移场景位姿")
    parser.add_argument("--source-root", type=Path, required=True, help="ACT 项目的 mode 目录")
    parser.add_argument("--tolerance", type=float, default=1e-9, help="最大允许绝对误差")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """执行空间布局等价性检查并打印 JSON 报告。

    Args:
        argv: 可选命令行参数；为空时读取当前进程参数。

    Returns:
        验证成功时返回0。
    """
    args = build_parser().parse_args(argv)
    report = verify_layout(args.source_root, args.tolerance)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
