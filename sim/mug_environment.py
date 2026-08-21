"""独立的ACT杯子双放置区MuJoCo环境。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from numpy.typing import NDArray

from sim.environment import CleanTabletopEnv


MUG_BODY_NAME = "body_obj_mug_5"
MUG_BOTTOM_SITE_NAME = "bottom_site_mug_5"
MUG_TOP_SITE_NAME = "top_site_mug_5"
MUG_PAD_BODY_NAMES = ("task_blue_pad", "task_yellow_pad")
MUG_PAD_GEOM_NAMES = ("task_blue_pad_geom", "task_yellow_pad_geom")
MUG_TASK_IDS = ("mug_on_blue", "mug_on_yellow")
MUG_TASK_PADS = {
    "mug_on_blue": ("task_blue_pad", "task_yellow_pad"),
    "mug_on_yellow": ("task_yellow_pad", "task_blue_pad"),
}
MUG_TEXTURE_ASSET_KEY = "mug_5/visual/image0.png"
MUG_APPEARANCE_TEXTURES = {
    "original": Path(MUG_TEXTURE_ASSET_KEY),
    "green_white": Path("mug_5/visual/image0_green_white.png"),
}
MUG_SAMPLE_X_RANGE = (0.25, 0.39)
MUG_SAMPLE_Y_RANGE = (-0.35, 0.35)
MUG_INITIAL_Z = 0.86
MUG_SETTLE_STEPS = 250
MUG_MAX_RESET_ATTEMPTS = 100
MUG_FOOTPRINT_RADIUS = 0.071
MUG_INITIAL_PAD_CLEARANCE = 0.01
PAD_HALF_SIZE = (0.08, 0.06, 0.001)
PLACEMENT_INSET = 0.01
STABLE_LINEAR_SPEED = 0.02
STABLE_ANGULAR_SPEED = 0.2
STABLE_DURATION_SECONDS = 0.5


@dataclass(frozen=True)
class MugSceneSnapshot:
    """描述一次可复现的杯子场景重置结果。

    Attributes:
        scene_seed: 生成杯子平面位置的随机种子。
        mug_initial_pose: 稳定后的杯子 ``xyz+quaternion``，形状为 ``(7,)``。
        pad_positions: 蓝、黄放置区的固定世界坐标，形状为 ``(2, 3)``。
    """

    scene_seed: int
    mug_initial_pose: NDArray[np.float64]
    pad_positions: NDArray[np.float64]


@dataclass(frozen=True)
class MugTaskEvaluation:
    """描述杯子放置任务的严格成功状态。

    Attributes:
        success: 是否满足目标区域、稳定和松爪条件。
        failure_mode: 当前成功状态或明确失败类别。
        metrics: 用于采集质检和调试的数值指标。
    """

    success: bool
    failure_mode: str
    metrics: dict[str, float | bool]


def resolve_mug_texture_path(asset_root: Path, appearance_variant: str) -> Path:
    """解析杯子外观变体对应的纹理路径。

    Args:
        asset_root: 包含杯子资源的MuJoCo资源根目录。
        appearance_variant: ``original``或``green_white``。

    Returns:
        已确认存在的纹理绝对路径。

    Raises:
        ValueError: 外观变体不受支持时抛出。
        FileNotFoundError: 变体对应的纹理不存在时抛出。
    """
    relative_path = MUG_APPEARANCE_TEXTURES.get(appearance_variant)
    if relative_path is None:
        supported = ", ".join(sorted(MUG_APPEARANCE_TEXTURES))
        raise ValueError(f"未知杯子外观变体: {appearance_variant!r}，可选值: {supported}")
    texture_path = (asset_root / relative_path).resolve()
    if not texture_path.is_file():
        raise FileNotFoundError(f"杯子外观纹理不存在: {texture_path}")
    return texture_path


def _load_mug_model(
    asset_root: Path,
    appearance_variant: str = "original",
) -> mujoco.MjModel:
    """从项目内存资源包加载独立杯子场景。

    Windows上的MuJoCo原生资源接口可能无法稳定处理中文绝对路径，因此
    Python先读取XML、纹理和mesh，再以相对资源键交给MuJoCo编译。杯子
    目录递归加入资源包，确保视觉网格和32个碰撞网格都能被解析。

    Args:
        asset_root: 包含 ``mug_scene.xml`` 的MuJoCo资源根目录。
        appearance_variant: 杯子纹理变体；默认保留原始红白纹理。

    Returns:
        编译完成的杯子场景模型。

    Raises:
        FileNotFoundError: 主XML或任一必要资源不存在时抛出。
    """
    main_xml_path = asset_root / "mug_scene.xml"
    appearance_texture_path = resolve_mug_texture_path(asset_root, appearance_variant)
    required_files = [
        asset_root / "ur10e_with_2f85_d435i.xml",
        asset_root / "tabletop" / "object" / "object_table.xml",
        asset_root / "tabletop" / "mesh" / "light_wood_v3.png",
    ]
    for directory in (
        asset_root / "universal_robots_ur10e" / "assets",
        asset_root / "robotiq_2f85" / "assets",
        asset_root / "realsense_d435i" / "assets",
    ):
        if directory.is_dir():
            required_files.extend(sorted(path for path in directory.iterdir() if path.is_file()))

    mug_root = asset_root / "mug_5"
    if mug_root.is_dir():
        required_files.extend(sorted(path for path in mug_root.rglob("*") if path.is_file()))

    missing = [path for path in [main_xml_path, *required_files] if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"杯子MuJoCo场景资源不完整: {missing}")

    assets = {
        path.relative_to(asset_root).as_posix(): path.read_bytes()
        for path in required_files
    }
    # XML始终引用原始资源键；只替换内存中的纹理字节，避免复制或修改场景XML。
    assets[MUG_TEXTURE_ASSET_KEY] = appearance_texture_path.read_bytes()
    return mujoco.MjModel.from_xml_string(
        main_xml_path.read_text(encoding="utf-8"),
        assets=assets,
    )


class MugTabletopEnv(CleanTabletopEnv):
    """展示UR10e、可交互杯子和两个放置区域的独立环境。

    该类继承原积木环境中已经验证的机器人控制、相机渲染、空间布局和
    MuJoCo内部Viewer能力，但使用独立模型、重置状态和任务判定，不修改
    ``CleanTabletopEnv``或原双积木场景。

    Args:
        project_root: SmolVLA项目根目录；默认由当前文件位置自动推导。
        image_size: 固定相机输出的 ``(宽, 高)``。
        appearance_variant: 杯子外观变体；默认使用原始红白纹理。
    """

    def __init__(
        self,
        project_root: Path | None = None,
        image_size: tuple[int, int] = (256, 256),
        appearance_variant: str = "original",
    ) -> None:
        """加载独立杯子模型并建立初始稳定场景。

        Args:
            project_root: SmolVLA项目根目录；为空时使用 ``sim`` 的父目录。
            image_size: 固定相机输出的 ``(宽, 高)``。
            appearance_variant: ``original``或``green_white``。

        Raises:
            FileNotFoundError: 杯子场景XML不存在时抛出。
            ValueError: 图像尺寸不是正整数时抛出。
        """
        self.project_root = (project_root or Path(__file__).resolve().parents[1]).resolve()
        self.scene_relative_path = Path("assets") / "mujoco" / "mug_scene.xml"
        self.scene_path = self.project_root / self.scene_relative_path
        if not self.scene_path.is_file():
            raise FileNotFoundError(f"找不到杯子MuJoCo主场景: {self.scene_path}")

        width, height = image_size
        if width <= 0 or height <= 0:
            raise ValueError(f"图像尺寸必须为正整数，实际为 {image_size}")
        self.image_size = (int(width), int(height))

        self.appearance_variant = appearance_variant
        self.appearance_texture_path = resolve_mug_texture_path(
            self.scene_path.parent,
            appearance_variant,
        )
        self.model = _load_mug_model(self.scene_path.parent, appearance_variant)
        self.data = mujoco.MjData(self.model)
        self._renderer: mujoco.Renderer | None = None
        self._mug_snapshot: MugSceneSnapshot | None = None
        self._mug_stable_since: dict[str, float | None] = {
            task_id: None for task_id in MUG_TASK_IDS
        }
        self.reset(scene_seed=0)

    def _mug_ids(self) -> tuple[int, int]:
        """返回杯子根body及其free joint编号。

        ACT的杯子XML没有给free joint命名，因此通过根body的 ``body_jntadr``
        查询，既避免修改源模型关节结构，也能稳定定位七维自由关节。

        Returns:
            ``(body_id, joint_id)``。

        Raises:
            ValueError: 杯子不存在、关节数不为1或不是free joint时抛出。
        """
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, MUG_BODY_NAME)
        if body_id < 0:
            raise ValueError(f"杯子场景缺少body: {MUG_BODY_NAME}")
        if int(self.model.body_jntnum[body_id]) != 1:
            raise ValueError("杯子根body必须恰好包含一个free joint")
        joint_id = int(self.model.body_jntadr[body_id])
        if self.model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError("杯子根body的唯一关节不是free joint")
        return body_id, joint_id

    def _body_position(self, body_name: str) -> NDArray[np.float64]:
        """读取指定body的世界位置副本。

        Args:
            body_name: XML中的body名称。

        Returns:
            三维世界位置。

        Raises:
            ValueError: body不存在时抛出。
        """
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise ValueError(f"杯子场景缺少body: {body_name}")
        return self.data.xpos[body_id].copy()

    def _mug_pose(self) -> NDArray[np.float64]:
        """读取杯子的七维世界位姿。

        Returns:
            ``xyz+quaternion``浮点数组。
        """
        body_id, _ = self._mug_ids()
        return np.concatenate([self.data.xpos[body_id], self.data.xquat[body_id]]).astype(
            np.float64
        )

    def _mug_speed(self) -> tuple[float, float]:
        """读取杯子free joint的线速度和角速度范数。

        Returns:
            ``(线速度范数, 角速度范数)``。
        """
        _, joint_id = self._mug_ids()
        dof_address = int(self.model.jnt_dofadr[joint_id])
        velocity = self.data.qvel[dof_address:dof_address + 6]
        return float(np.linalg.norm(velocity[:3])), float(np.linalg.norm(velocity[3:]))

    def _too_close_to_pad(self, xy: NDArray[np.float64]) -> bool:
        """判断候选杯子是否与任一放置区视觉范围过近。

        Args:
            xy: 杯子候选根body平面坐标。

        Returns:
            考虑杯子把手半径和1厘米间隔后发生重叠时返回 ``True``。
        """
        expanded_half_size = np.asarray(PAD_HALF_SIZE[:2]) + (
            MUG_FOOTPRINT_RADIUS + MUG_INITIAL_PAD_CLEARANCE
        )
        return any(
            bool(np.all(np.abs(xy - self._body_position(pad_name)[:2]) <= expanded_half_size))
            for pad_name in MUG_PAD_BODY_NAMES
        )

    def _has_mug_robot_contact(self) -> bool:
        """检查当前接触中是否存在杯子与机器人直接接触。

        Returns:
            杯子和机器人根运动树同时出现在某个接触中时返回 ``True``。
        """
        mug_body_id, _ = self._mug_ids()
        robot_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base")
        mug_root_id = int(self.model.body_rootid[mug_body_id])
        robot_root_id = int(self.model.body_rootid[robot_body_id])
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            contact_roots = {
                int(self.model.body_rootid[self.model.geom_bodyid[contact.geom1]]),
                int(self.model.body_rootid[self.model.geom_bodyid[contact.geom2]]),
            }
            if {mug_root_id, robot_root_id}.issubset(contact_roots):
                return True
        return False

    def _settled_scene_error(self) -> str:
        """检查稳定后的杯子场景是否满足reset验收条件。

        Returns:
            空字符串表示合法，否则返回稳定的拒绝原因标识。
        """
        pose = self._mug_pose()
        if not np.isfinite(pose).all() or not np.isfinite(self.data.qvel).all():
            return "non_finite_state"
        if self._has_mug_robot_contact():
            return "mug_robot_contact"
        x, y, z = pose[:3]
        if not (0.20 <= x <= 0.44 and -0.44 <= y <= 0.44 and 0.80 <= z <= 0.90):
            return "mug_out_of_bounds"
        return ""

    def reset(self, scene_seed: int = 0) -> MugSceneSnapshot:
        """按显式种子随机杯子位置并等待其稳定落在桌面上。

        每次候选都从干净动力学状态开始。杯子以单位四元数从 ``z=0.86``
        落下并推进250个物理步；只有位置、接触和数值状态均合法的候选才会
        成为episode初始状态。稳定过程结束后仿真时间重新归零。

        Args:
            scene_seed: 杯子平面位置采样使用的整数种子。

        Returns:
            稳定后的杯子位姿和固定放置区坐标快照。

        Raises:
            ValueError: ``scene_seed``不是整数时抛出。
            RuntimeError: 连续100次都无法生成合法场景时抛出。
        """
        if isinstance(scene_seed, bool) or not isinstance(scene_seed, (int, np.integer)):
            raise ValueError(f"scene_seed必须为整数，实际为 {scene_seed!r}")

        rng = np.random.default_rng(int(scene_seed))
        rejection_counts: dict[str, int] = {}
        _, mug_joint_id = self._mug_ids()
        qpos_address = int(self.model.jnt_qposadr[mug_joint_id])
        for _ in range(MUG_MAX_RESET_ATTEMPTS):
            mujoco.mj_resetData(self.model, self.data)
            self._reset_robot()
            mug_xy = rng.uniform(
                low=[MUG_SAMPLE_X_RANGE[0], MUG_SAMPLE_Y_RANGE[0]],
                high=[MUG_SAMPLE_X_RANGE[1], MUG_SAMPLE_Y_RANGE[1]],
            )
            mujoco.mj_forward(self.model, self.data)
            if self._too_close_to_pad(mug_xy):
                rejection_counts["pad_clearance"] = rejection_counts.get("pad_clearance", 0) + 1
                continue

            self.data.qpos[qpos_address:qpos_address + 7] = (
                float(mug_xy[0]),
                float(mug_xy[1]),
                MUG_INITIAL_Z,
                1.0,
                0.0,
                0.0,
                0.0,
            )
            mujoco.mj_forward(self.model, self.data)
            if self._has_mug_robot_contact():
                rejection_counts["mug_robot_contact"] = (
                    rejection_counts.get("mug_robot_contact", 0) + 1
                )
                continue

            for _ in range(MUG_SETTLE_STEPS):
                mujoco.mj_step(self.model, self.data)
            scene_error = self._settled_scene_error()
            if scene_error:
                rejection_counts[scene_error] = rejection_counts.get(scene_error, 0) + 1
                continue
            break
        else:
            raise RuntimeError(
                f"杯子随机化失败: scene_seed={scene_seed}, "
                f"attempts={MUG_MAX_RESET_ATTEMPTS}, rejections={rejection_counts}"
            )

        # 稳定杯子时机械臂position actuator会产生极小的跟随误差。episode
        # 开始前精确恢复ACT关节角并清零速度，保证腕部相机与原场景位姿一致；
        # 此时杯子已经静止，因此清零其残余数值速度不会改变稳定位置。
        self._reset_robot()
        self.data.qvel[:] = 0.0
        self.data.time = 0.0
        mujoco.mj_forward(self.model, self.data)
        pad_positions = np.stack(
            [self._body_position(body_name) for body_name in MUG_PAD_BODY_NAMES]
        )
        self._mug_snapshot = MugSceneSnapshot(
            scene_seed=int(scene_seed),
            mug_initial_pose=self._mug_pose().copy(),
            pad_positions=pad_positions.copy(),
        )
        self._mug_stable_since = {task_id: None for task_id in MUG_TASK_IDS}
        return self.scene_snapshot()

    def scene_snapshot(self) -> MugSceneSnapshot:
        """返回当前杯子场景快照的防御性副本。

        Returns:
            当前reset对应的不可变快照。

        Raises:
            RuntimeError: 环境尚未完成reset时抛出。
        """
        if self._mug_snapshot is None:
            raise RuntimeError("杯子环境尚未完成reset")
        return MugSceneSnapshot(
            scene_seed=self._mug_snapshot.scene_seed,
            mug_initial_pose=self._mug_snapshot.mug_initial_pose.copy(),
            pad_positions=self._mug_snapshot.pad_positions.copy(),
        )

    def _site_position(self, site_name: str) -> NDArray[np.float64]:
        """读取指定site的世界位置。

        Args:
            site_name: XML中的site名称。

        Returns:
            三维世界位置副本。

        Raises:
            ValueError: site不存在时抛出。
        """
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id < 0:
            raise ValueError(f"杯子场景缺少site: {site_name}")
        return self.data.site_xpos[site_id].copy()

    def task_layout(self) -> dict[str, dict[str, NDArray[np.float64] | float | int | str]]:
        """返回杯子和蓝黄区域的只读状态快照。

        Returns:
            以body名称为键的属性字典。杯子项包含位姿、质量、速度和上下
            边界site；区域项包含位置、尺寸、颜色及碰撞掩码。数组均为副本。
        """
        mug_body_id, _ = self._mug_ids()
        linear_speed, angular_speed = self._mug_speed()
        layout: dict[str, dict[str, NDArray[np.float64] | float | int | str]] = {
            MUG_BODY_NAME: {
                "kind": "mug",
                "position": self.data.xpos[mug_body_id].copy(),
                "quaternion": self.data.xquat[mug_body_id].copy(),
                "mass": float(self.model.body_subtreemass[mug_body_id]),
                "linear_speed": linear_speed,
                "angular_speed": angular_speed,
                "bottom_site_position": self._site_position(MUG_BOTTOM_SITE_NAME),
                "top_site_position": self._site_position(MUG_TOP_SITE_NAME),
            }
        }
        for body_name, geom_name in zip(
            MUG_PAD_BODY_NAMES,
            MUG_PAD_GEOM_NAMES,
            strict=True,
        ):
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            layout[body_name] = {
                "kind": "pad",
                "position": self.data.xpos[body_id].copy(),
                "geom_half_size": self.model.geom_size[geom_id].copy(),
                "rgba": self.model.geom_rgba[geom_id].copy(),
                "mass": float(self.model.body_mass[body_id]),
                "contype": int(self.model.geom_contype[geom_id]),
                "conaffinity": int(self.model.geom_conaffinity[geom_id]),
            }
        return layout

    def _is_mug_inside_pad(self, pad_name: str) -> tuple[bool, bool, bool]:
        """判断杯子中心、落桌高度和直立方向是否满足目标区条件。

        Args:
            pad_name: 蓝色或黄色区域body名称。

        Returns:
            ``(中心在区域内, 高度正确, 保持直立)``。
        """
        mug_position = self._body_position(MUG_BODY_NAME)
        pad_position = self._body_position(pad_name)
        center_limit = np.asarray(PAD_HALF_SIZE[:2])
        center_inside = bool(np.all(np.abs(mug_position[:2] - pad_position[:2]) <= center_limit))
        bottom_position = self._site_position(MUG_BOTTOM_SITE_NAME)
        top_position = self._site_position(MUG_TOP_SITE_NAME)
        correct_height = abs(float(bottom_position[2]) - 0.8) <= 0.015
        upright = float(top_position[2] - bottom_position[2]) >= 0.07
        return center_inside, correct_height, upright

    def evaluate_task(
        self,
        task_id: str,
        elapsed_seconds: float | None = None,
        timeout_seconds: float | None = None,
    ) -> MugTaskEvaluation:
        """评估杯子到蓝区或黄区任务的严格成功状态。

        Args:
            task_id: ``mug_on_blue``或``mug_on_yellow``。
            elapsed_seconds: 当前episode已经执行的秒数。
            timeout_seconds: 可选超时阈值。

        Returns:
            成功状态、失败分类及关键数值指标。

        Raises:
            ValueError: 任务标识未知时抛出。
        """
        if task_id not in MUG_TASK_PADS:
            raise ValueError(f"未知task_id={task_id!r}，可选值={MUG_TASK_IDS}")
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            return MugTaskEvaluation(False, "control_exception", {"finite_state": False})

        target_pad, other_pad = MUG_TASK_PADS[task_id]
        center_inside, correct_height, upright = self._is_mug_inside_pad(target_pad)
        other_center_inside, other_height, other_upright = self._is_mug_inside_pad(other_pad)
        target_inside = center_inside and correct_height and upright
        inside_other = other_center_inside and other_height and other_upright
        linear_speed, angular_speed = self._mug_speed()
        stable_now = linear_speed < STABLE_LINEAR_SPEED and angular_speed < STABLE_ANGULAR_SPEED
        if target_inside and stable_now:
            if self._mug_stable_since[task_id] is None:
                self._mug_stable_since[task_id] = float(self.data.time)
        else:
            self._mug_stable_since[task_id] = None
        stable_duration = (
            0.0
            if self._mug_stable_since[task_id] is None
            else float(self.data.time) - float(self._mug_stable_since[task_id])
        )
        gripper_released = bool(self.get_state()[6] < 0.5)
        bottom_z = float(self._site_position(MUG_BOTTOM_SITE_NAME)[2])
        success = (
            target_inside
            and stable_duration >= STABLE_DURATION_SECONDS
            and gripper_released
        )

        mug_position = self._body_position(MUG_BODY_NAME)
        out_of_bounds = bool(
            mug_position[2] < 0.75
            or not 0.15 <= mug_position[0] <= 0.70
            or not -0.50 <= mug_position[1] <= 0.50
        )
        if success:
            failure_mode = "success"
        elif inside_other:
            failure_mode = "wrong_pad"
        elif out_of_bounds:
            failure_mode = "dropped_or_out_of_bounds"
        elif timeout_seconds is not None and elapsed_seconds is not None and elapsed_seconds >= timeout_seconds:
            failure_mode = "timeout"
        else:
            failure_mode = "in_progress"
        return MugTaskEvaluation(
            success=success,
            failure_mode=failure_mode,
            metrics={
                "center_inside": center_inside,
                "correct_height": correct_height,
                "upright": upright,
                "target_inside": target_inside,
                "bottom_z": bottom_z,
                "inside_other_pad": inside_other,
                "linear_speed": linear_speed,
                "angular_speed": angular_speed,
                "stable_duration": stable_duration,
                "gripper_released": gripper_released,
                "finite_state": True,
            },
        )
