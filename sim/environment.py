"""参考 ACT 演示布局构建的双积木 MuJoCo 环境封装。"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from numpy.typing import NDArray

from sim.mujoco_viewer import EmbeddedCameraViewer


ARM_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
ACT_INITIAL_ARM_QPOS_DEG = np.array([0.0, -90.0, 90.0, -90.0, -90.0, 90.0])
DISPLAY_CAMERA_NAMES = ("agentview", "d435i_rgb", "sideview")
TASK_CUBE_BODY_NAMES = ("task_red_cube", "task_green_cube")
TASK_CUBE_GEOM_NAMES = ("task_red_cube_geom", "task_green_cube_geom")
TASK_PAD_BODY_NAMES = ("task_blue_pad", "task_yellow_pad")
TASK_PAD_GEOM_NAMES = ("task_blue_pad_geom", "task_yellow_pad_geom")
CUBE_HALF_SIZE = (0.025, 0.025, 0.025)
PAD_HALF_SIZE = (0.08, 0.06, 0.001)
TASK_INITIAL_BODY_POSITIONS = (
    ("task_red_cube", (0.35, -0.22, 0.825)),
    ("task_green_cube", (0.35, 0.22, 0.825)),
    ("task_blue_pad", (0.55, -0.22, 0.8005)),
    ("task_yellow_pad", (0.55, 0.22, 0.8005)),
)
CUBE_SAMPLE_X_RANGE = (0.25, 0.42)
CUBE_SAMPLE_Y_RANGE = (-0.35, 0.35)
CUBE_MIN_CENTER_DISTANCE = 0.12
CUBE_INITIAL_Z = 0.825
MAX_RESET_ATTEMPTS = 100
TASK_IDS = (
    "red_on_blue",
    "red_on_yellow",
    "green_on_blue",
    "green_on_yellow",
)
TASK_OBJECTS = {
    "red_on_blue": ("task_red_cube", "task_green_cube", "task_blue_pad", "task_yellow_pad"),
    "red_on_yellow": ("task_red_cube", "task_green_cube", "task_yellow_pad", "task_blue_pad"),
    "green_on_blue": ("task_green_cube", "task_red_cube", "task_blue_pad", "task_yellow_pad"),
    "green_on_yellow": ("task_green_cube", "task_red_cube", "task_yellow_pad", "task_blue_pad"),
}
PLACEMENT_INSET = 0.005
STABLE_LINEAR_SPEED = 0.02
STABLE_ANGULAR_SPEED = 0.2
STABLE_DURATION_SECONDS = 0.5
GRIPPER_RELEASED_QPOS = 0.1
RgbImage = NDArray[np.uint8]


@dataclass(frozen=True)
class SceneSnapshot:
    """描述一次可复现 reset 后的任务场景。

    Attributes:
        scene_seed: 生成积木位置的随机种子。
        cube_initial_poses: 红、绿积木的 ``xyz+quaternion``，形状为 ``(2, 7)``。
        pad_positions: 蓝、黄放置区的固定世界坐标，形状为 ``(2, 3)``。
    """

    scene_seed: int
    cube_initial_poses: NDArray[np.float64]
    pad_positions: NDArray[np.float64]


@dataclass(frozen=True)
class TaskEvaluation:
    """返回严格成功状态和当前失败分类。

    Attributes:
        success: 是否满足完整严格成功条件。
        failure_mode: ``success``、``in_progress`` 或明确失败类别。
        metrics: 便于调试和质检的数值指标。
    """

    success: bool
    failure_mode: str
    metrics: dict[str, float | bool]


def _load_model_from_asset_bundle(asset_root: Path) -> mujoco.MjModel:
    """从 Python 读取的内存资源包编译 MuJoCo 模型。

    MuJoCo 3.6.0 的 Windows 原生文件接口无法稳定读取包含中文字符的资源
    绝对路径。Python 的 ``pathlib`` 可以正确读取这些文件，因此先把主 XML、
    include XML、mesh 和纹理读成内存，再用相对资源键交给 MuJoCo 编译。
    这种方式不会重写 XML，也不会改变任何位姿或物理参数。

    Args:
        asset_root: 包含 ``scene.xml`` 和全部场景资源的目录。

    Returns:
        编译完成的 MuJoCo 模型。

    Raises:
        FileNotFoundError: 任一必需资源不存在时抛出。
    """
    main_xml_path = asset_root / "scene.xml"
    required_files = [
        asset_root / "ur10e_with_2f85_d435i.xml",
        asset_root / "tabletop" / "object" / "object_table.xml",
        asset_root / "tabletop" / "mesh" / "light_wood_v3.png",
    ]
    mesh_directories = (
        asset_root / "universal_robots_ur10e" / "assets",
        asset_root / "robotiq_2f85" / "assets",
        asset_root / "realsense_d435i" / "assets",
    )
    for directory in mesh_directories:
        required_files.extend(sorted(path for path in directory.iterdir() if path.is_file()))

    missing = [path for path in [main_xml_path, *required_files] if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"MuJoCo 场景资源不完整: {missing}")

    # MuJoCo 用资源键解析 include/file 属性；只传运行闭包，避免多个 LICENSE
    # 或 README 同名文件被原生编译器判定为重复资源。
    assets = {
        path.relative_to(asset_root).as_posix(): path.read_bytes()
        for path in required_files
    }
    main_xml = main_xml_path.read_text(encoding="utf-8")
    return mujoco.MjModel.from_xml_string(main_xml, assets=assets)


def _require_object_id(
    model: mujoco.MjModel,
    object_type: mujoco.mjtObj,
    name: str,
) -> int:
    """按名称查询 MuJoCo 对象编号，并在缺失时给出明确错误。

    Args:
        model: 已加载的 MuJoCo 模型。
        object_type: MuJoCo 对象类型，例如 body、geom 或 camera。
        name: XML 中定义的对象名称。

    Returns:
        非负的 MuJoCo 对象编号。

    Raises:
        ValueError: 模型中不存在指定对象时抛出。
    """
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise ValueError(f"MuJoCo 场景缺少对象: type={object_type.name}, name={name}")
    return object_id


class CleanTabletopEnv:
    """展示 ACT UR10e、双积木与放置区域的 MuJoCo 环境。

    该类提供ACT初始关节姿态、确定性任务布局、仿真推进、固定相机取图和
    交互式Viewer。积木随机化、任务成功判定和遥操作控制不属于本阶段职责。

    Args:
        project_root: SmolVLA 项目根目录。默认由本文件位置自动推导。
        image_size: 固定相机输出的 ``(宽, 高)``，默认均为 256。
    """

    def __init__(
        self,
        project_root: Path | None = None,
        image_size: tuple[int, int] = (256, 256),
    ) -> None:
        """加载场景并初始化仿真状态。

        Args:
            project_root: SmolVLA 项目根目录；为空时自动使用 ``sim`` 的父目录。
            image_size: 固定相机输出的 ``(宽, 高)``。

        Raises:
            FileNotFoundError: 主场景 XML 不存在时抛出。
            ValueError: 图像尺寸不是正整数时抛出。
        """
        self.project_root = (project_root or Path(__file__).resolve().parents[1]).resolve()
        self.scene_relative_path = Path("assets") / "mujoco" / "scene.xml"
        self.scene_path = self.project_root / self.scene_relative_path
        if not self.scene_path.is_file():
            raise FileNotFoundError(f"找不到 MuJoCo 主场景: {self.scene_path}")

        width, height = image_size
        if width <= 0 or height <= 0:
            raise ValueError(f"图像尺寸必须为正整数，实际为 {image_size}")
        self.image_size = (int(width), int(height))

        self.model = _load_model_from_asset_bundle(self.scene_path.parent)
        self.data = mujoco.MjData(self.model)
        self._renderer: mujoco.Renderer | None = None
        self._scene_snapshot: SceneSnapshot | None = None
        self._stable_since: dict[str, float | None] = {task_id: None for task_id in TASK_IDS}
        self.reset(scene_seed=0)

    def __enter__(self) -> "CleanTabletopEnv":
        """返回当前环境，支持 ``with`` 自动资源管理。

        Returns:
            当前环境实例。
        """
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        """退出上下文时释放渲染资源。

        Args:
            exc_type: 上下文内部异常类型，没有异常时为 ``None``。
            exc_value: 上下文内部异常对象。
            traceback: 上下文内部异常堆栈。
        """
        del exc_type, exc_value, traceback
        self.close()

    def reset(self, scene_seed: int = 0) -> SceneSnapshot:
        """按显式种子随机化积木并恢复机器人初始状态。

        蓝、黄放置区保持 XML 中的固定坐标。红、绿积木只随机平面位置，
        姿态始终为单位四元数，且中心距离至少为12厘米。

        Args:
            scene_seed: 用于本次积木位置采样的整数种子。

        Returns:
            包含种子、积木初始位姿和固定放置区坐标的只读快照。

        Raises:
            RuntimeError: 连续100次采样都无法生成合法场景时抛出。
        """
        if isinstance(scene_seed, bool) or not isinstance(scene_seed, (int, np.integer)):
            raise ValueError(f"scene_seed 必须为整数，实际为 {scene_seed!r}")

        rng = np.random.default_rng(int(scene_seed))
        rejection_counts: dict[str, int] = {}
        for _ in range(MAX_RESET_ATTEMPTS):
            mujoco.mj_resetData(self.model, self.data)
            self._reset_robot()
            cube_xy = rng.uniform(
                low=[CUBE_SAMPLE_X_RANGE[0], CUBE_SAMPLE_Y_RANGE[0]],
                high=[CUBE_SAMPLE_X_RANGE[1], CUBE_SAMPLE_Y_RANGE[1]],
                size=(2, 2),
            )
            if float(np.linalg.norm(cube_xy[0] - cube_xy[1])) < CUBE_MIN_CENTER_DISTANCE:
                rejection_counts["cube_clearance"] = rejection_counts.get("cube_clearance", 0) + 1
                continue

            for joint_name, xy in zip(
                ("task_red_cube_free_joint", "task_green_cube_free_joint"),
                cube_xy,
                strict=True,
            ):
                joint_id = _require_object_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                qpos_address = int(self.model.jnt_qposadr[joint_id])
                self.data.qpos[qpos_address:qpos_address + 7] = (
                    float(xy[0]),
                    float(xy[1]),
                    CUBE_INITIAL_Z,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                )
            mujoco.mj_forward(self.model, self.data)
            contact_error = self._initial_contact_error()
            if contact_error:
                rejection_counts[contact_error] = rejection_counts.get(contact_error, 0) + 1
                continue
            break
        else:
            raise RuntimeError(
                f"积木随机化失败: scene_seed={scene_seed}, "
                f"attempts={MAX_RESET_ATTEMPTS}, rejections={rejection_counts}"
            )

        cube_poses = np.stack([self._body_pose(name) for name in TASK_CUBE_BODY_NAMES])
        pad_positions = np.stack(
            [self.task_layout()[name]["position"] for name in TASK_PAD_BODY_NAMES]
        )
        self._scene_snapshot = SceneSnapshot(
            scene_seed=int(scene_seed),
            cube_initial_poses=cube_poses.copy(),
            pad_positions=pad_positions.copy(),
        )
        self._stable_since = {task_id: None for task_id in TASK_IDS}
        return self.scene_snapshot()

    def _reset_robot(self) -> None:
        """恢复机械臂、夹爪控制量和初始关节角。"""
        initial_qpos = np.deg2rad(ACT_INITIAL_ARM_QPOS_DEG)
        for joint_name, joint_value in zip(ARM_JOINT_NAMES, initial_qpos, strict=True):
            joint_id = _require_object_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            self.data.qpos[int(self.model.jnt_qposadr[joint_id])] = joint_value
        self.data.ctrl[:6] = initial_qpos
        self.data.ctrl[6:] = 0.0

    def _initial_contact_error(self) -> str:
        """检查积木之间以及积木与机器人之间的非法初始接触。

        Returns:
            空字符串表示合法，否则返回拒绝原因。
        """
        cube_body_ids = {
            _require_object_id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in TASK_CUBE_BODY_NAMES
        }
        robot_root_id = _require_object_id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base")
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            roots = {
                int(self.model.body_rootid[self.model.geom_bodyid[contact.geom1]]),
                int(self.model.body_rootid[self.model.geom_bodyid[contact.geom2]]),
            }
            if roots == cube_body_ids:
                return "cube_contact"
            if robot_root_id in roots and roots.intersection(cube_body_ids):
                return "cube_robot_contact"
        return ""

    def _body_pose(self, body_name: str) -> NDArray[np.float64]:
        """读取指定 body 的世界位置和四元数。

        Args:
            body_name: MuJoCo body 名称。

        Returns:
            ``xyz+quaternion`` 七维数组。
        """
        body_id = _require_object_id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        return np.concatenate([self.data.xpos[body_id], self.data.xquat[body_id]]).astype(np.float64)

    def scene_snapshot(self) -> SceneSnapshot:
        """返回当前场景快照的防御性副本。

        Returns:
            当前 reset 对应的场景快照。
        """
        if self._scene_snapshot is None:
            raise RuntimeError("环境尚未完成 reset")
        return SceneSnapshot(
            scene_seed=self._scene_snapshot.scene_seed,
            cube_initial_poses=self._scene_snapshot.cube_initial_poses.copy(),
            pad_positions=self._scene_snapshot.pad_positions.copy(),
        )

    def step(self, steps: int = 1) -> None:
        """推进指定数量的 MuJoCo 物理步。

        Args:
            steps: 连续执行的物理步数，必须大于零。

        Raises:
            ValueError: ``steps`` 小于1时抛出。
            RuntimeError: 仿真状态出现 NaN 或无穷值时抛出。
        """
        if steps < 1:
            raise ValueError(f"steps 必须大于零，实际为 {steps}")
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise RuntimeError("MuJoCo 仿真状态出现非有限值")

    def capture_cameras(self) -> "OrderedDict[str, RgbImage]":
        """渲染 ACT 展示使用的前视、腕部和侧视三路 RGB 图像。

        Returns:
            按 ``agentview``、``d435i_rgb``、``sideview`` 排列的图像映射。
            每张图像均为 ``256×256×3`` 的 RGB uint8 数组。
        """
        width, height = self.image_size
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=height, width=width)

        images: "OrderedDict[str, RgbImage]" = OrderedDict()
        for camera_name in DISPLAY_CAMERA_NAMES:
            _require_object_id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
            self._renderer.update_scene(self.data, camera=camera_name)
            images[camera_name] = self._renderer.render().copy()
        return images

    def capture_training_images(self) -> "OrderedDict[str, RgbImage]":
        """渲染策略训练使用的第三方和腕部两路RGB图像。

        Returns:
            按 ``agent``、``wrist`` 排列的两张 ``256×256×3`` RGB图像。
        """
        images = self.capture_cameras()
        return OrderedDict(
            (
                ("agent", images["agentview"]),
                ("wrist", images["d435i_rgb"]),
            )
        )

    def get_state(self) -> NDArray[np.float32]:
        """返回六个当前关节角和一个当前夹爪状态。

        Returns:
            七维 ``float32`` 状态，夹爪打开为0、闭合为1。
        """
        joint_positions = []
        for joint_name in ARM_JOINT_NAMES:
            joint_id = _require_object_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            joint_positions.append(self.data.qpos[int(self.model.jnt_qposadr[joint_id])])
        gripper_joint_id = _require_object_id(
            self.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            "right_driver_joint",
        )
        gripper_qpos = float(self.data.qpos[int(self.model.jnt_qposadr[gripper_joint_id])])
        gripper_state = 1.0 if gripper_qpos >= GRIPPER_RELEASED_QPOS else 0.0
        return np.asarray([*joint_positions, gripper_state], dtype=np.float32)

    def apply_joint_action(
        self,
        action: NDArray[np.floating],
        physics_steps: int = 0,
    ) -> NDArray[np.float32]:
        """设置七维绝对关节目标动作并可选推进仿真。

        Args:
            action: 六个绝对关节目标角和一个0/1夹爪指令。
            physics_steps: 设置控制量后需要推进的物理步数。

        Returns:
            执行后的七维当前状态。

        Raises:
            ValueError: 动作shape、数值或夹爪范围不合法时抛出。
        """
        command = np.asarray(action, dtype=np.float64)
        if command.shape != (7,) or not np.isfinite(command).all():
            raise ValueError(f"action 必须是有限七维向量，实际 shape={command.shape}")
        if not 0.0 <= command[6] <= 1.0:
            raise ValueError(f"夹爪指令必须位于[0,1]，实际为 {command[6]}")
        arm_ranges = self.model.actuator_ctrlrange[:6]
        if np.any(command[:6] < arm_ranges[:, 0]) or np.any(command[:6] > arm_ranges[:, 1]):
            raise ValueError("机械臂目标角超出 actuator ctrlrange")
        self.data.ctrl[:6] = command[:6]
        self.data.ctrl[6] = command[6] * 255.0
        if physics_steps:
            self.step(physics_steps)
        return self.get_state()

    def _is_cube_inside_pad(self, cube_name: str, pad_name: str) -> bool:
        """判断积木是否完整位于放置区内缩边界。

        Args:
            cube_name: 目标积木 body 名称。
            pad_name: 目标放置区 body 名称。

        Returns:
            水平边界和高度同时满足时返回 ``True``。
        """
        layout = self.task_layout()
        cube_position = np.asarray(layout[cube_name]["position"])
        pad_position = np.asarray(layout[pad_name]["position"])
        xy_limit = np.asarray(PAD_HALF_SIZE[:2]) - np.asarray(CUBE_HALF_SIZE[:2]) - PLACEMENT_INSET
        inside_xy = bool(np.all(np.abs(cube_position[:2] - pad_position[:2]) <= xy_limit))
        correct_height = abs(float(cube_position[2]) - CUBE_INITIAL_Z) <= 0.01
        return inside_xy and correct_height

    def _cube_speed(self, cube_name: str) -> tuple[float, float]:
        """读取积木自由关节的线速度和角速度范数。

        Args:
            cube_name: 红或绿积木 body 名称。

        Returns:
            ``(线速度范数, 角速度范数)``。
        """
        joint_id = _require_object_id(
            self.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            f"{cube_name}_free_joint",
        )
        dof_address = int(self.model.jnt_dofadr[joint_id])
        velocity = self.data.qvel[dof_address:dof_address + 6]
        return float(np.linalg.norm(velocity[:3])), float(np.linalg.norm(velocity[3:]))

    def evaluate_task(
        self,
        task_id: str,
        elapsed_seconds: float | None = None,
        timeout_seconds: float | None = None,
    ) -> TaskEvaluation:
        """评估四类语言任务的严格成功状态和失败类别。

        Args:
            task_id: ``red_on_blue`` 等四个固定任务标识之一。
            elapsed_seconds: 当前episode已经执行的秒数。
            timeout_seconds: 可选超时阈值。

        Returns:
            严格成功状态、失败类别和关键数值指标。

        Raises:
            ValueError: 任务标识未知时抛出。
        """
        if task_id not in TASK_OBJECTS:
            raise ValueError(f"未知 task_id={task_id!r}，可选值={TASK_IDS}")
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            return TaskEvaluation(False, "control_exception", {"finite_state": False})

        target_cube, other_cube, target_pad, other_pad = TASK_OBJECTS[task_id]
        target_inside = self._is_cube_inside_pad(target_cube, target_pad)
        other_inside_target = self._is_cube_inside_pad(other_cube, target_pad)
        target_inside_other = self._is_cube_inside_pad(target_cube, other_pad)
        linear_speed, angular_speed = self._cube_speed(target_cube)
        stable_now = linear_speed < STABLE_LINEAR_SPEED and angular_speed < STABLE_ANGULAR_SPEED
        if target_inside and stable_now:
            if self._stable_since[task_id] is None:
                self._stable_since[task_id] = float(self.data.time)
        else:
            self._stable_since[task_id] = None
        stable_duration = (
            0.0
            if self._stable_since[task_id] is None
            else float(self.data.time) - float(self._stable_since[task_id])
        )
        gripper_released = bool(self.get_state()[6] < 0.5)
        success = (
            target_inside
            and not other_inside_target
            and stable_duration >= STABLE_DURATION_SECONDS
            and gripper_released
        )

        target_position = np.asarray(self.task_layout()[target_cube]["position"])
        out_of_bounds = bool(
            target_position[2] < 0.75
            or not 0.15 <= target_position[0] <= 0.70
            or not -0.50 <= target_position[1] <= 0.50
        )
        if success:
            failure_mode = "success"
        elif other_inside_target:
            failure_mode = "wrong_cube"
        elif target_inside_other:
            failure_mode = "wrong_pad"
        elif out_of_bounds:
            failure_mode = "dropped_or_out_of_bounds"
        elif timeout_seconds is not None and elapsed_seconds is not None and elapsed_seconds >= timeout_seconds:
            failure_mode = "timeout"
        else:
            failure_mode = "in_progress"
        return TaskEvaluation(
            success,
            failure_mode,
            {
                "target_inside": target_inside,
                "other_inside_target": other_inside_target,
                "target_inside_other": target_inside_other,
                "linear_speed": linear_speed,
                "angular_speed": angular_speed,
                "stable_duration": stable_duration,
                "gripper_released": gripper_released,
                "finite_state": True,
            },
        )

    def spatial_layout(self) -> dict[str, NDArray[np.float64] | float]:
        """返回机械臂与桌面的编译后世界位姿和相对位置。

        Returns:
            包含机械臂基座、桌面 body、桌面 geom、相对平移、旋转矩阵、
            桌面半尺寸和桌面上表面高度的字典。
        """
        base_id = _require_object_id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base")
        table_body_id = _require_object_id(self.model, mujoco.mjtObj.mjOBJ_BODY, "front_object_table")
        table_geom_id = _require_object_id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "front_object_table")

        base_position = self.data.xpos[base_id].copy()
        table_body_position = self.data.xpos[table_body_id].copy()
        table_geom_position = self.data.geom_xpos[table_geom_id].copy()
        table_half_size = self.model.geom_size[table_geom_id].copy()
        return {
            "base_position": base_position,
            "base_rotation": self.data.xmat[base_id].reshape(3, 3).copy(),
            "table_body_position": table_body_position,
            "table_body_rotation": self.data.xmat[table_body_id].reshape(3, 3).copy(),
            "table_geom_position": table_geom_position,
            "table_half_size": table_half_size,
            "base_from_table_body": base_position - table_body_position,
            "base_from_table_geom": base_position - table_geom_position,
            "table_top_z": float(table_geom_position[2] + table_half_size[2]),
        }

    def task_layout(self) -> dict[str, dict[str, NDArray[np.float64] | float | int]]:
        """返回任务积木和放置区域的只读状态快照。

        返回值中的数组均为副本，调用方修改它们不会影响 MuJoCo 模型或当前
        仿真数据。该接口只公开布局和物理属性，不实现随机化或成功判定。

        Returns:
            以任务 body 名称为键的属性字典。每项包含世界位置、geom半尺寸、
            颜色、质量、摩擦和碰撞掩码。
        """
        task_pairs = zip(
            (*TASK_CUBE_BODY_NAMES, *TASK_PAD_BODY_NAMES),
            (*TASK_CUBE_GEOM_NAMES, *TASK_PAD_GEOM_NAMES),
            strict=True,
        )
        layout: dict[str, dict[str, NDArray[np.float64] | float | int]] = {}
        for body_name, geom_name in task_pairs:
            body_id = _require_object_id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            geom_id = _require_object_id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            layout[body_name] = {
                "position": self.data.xpos[body_id].copy(),
                "geom_half_size": self.model.geom_size[geom_id].copy(),
                "rgba": self.model.geom_rgba[geom_id].copy(),
                "mass": float(self.model.body_mass[body_id]),
                "friction": self.model.geom_friction[geom_id].copy(),
                "contype": int(self.model.geom_contype[geom_id]),
                "conaffinity": int(self.model.geom_conaffinity[geom_id]),
            }
        return layout

    def run(
        self,
        max_seconds: float | None = None,
        display_hz: float = 60.0,
        show_camera_panel: bool = True,
    ) -> dict[str, float]:
        """打开交互式 Viewer，并持续刷新场景和三路相机面板。

        主视角和三路固定相机使用同一个 ``MjrContext``，直接渲染到同一
        GLFW窗口，避免OpenCV窗口、GPU到CPU读回和跨上下文切换。显示默认
        为60 Hz，物理仿真仍按模型定义的500 Hz累计推进。

        Args:
            max_seconds: 可选的最长展示秒数；默认持续运行到用户关闭窗口。
            display_hz: MuJoCo主Viewer的目标刷新频率。
            show_camera_panel: 是否在MuJoCo窗口内显示三路固定相机。

        Returns:
            包含实际显示时长、渲染帧数和平均FPS的统计信息。

        Raises:
            ValueError: ``max_seconds`` 不为空且不大于零时抛出。
        """
        if max_seconds is not None and max_seconds <= 0:
            raise ValueError(f"max_seconds 必须大于零，实际为 {max_seconds}")
        if display_hz <= 0:
            raise ValueError(f"显示频率必须大于零，实际 display_hz={display_hz}")

        frame_period = 1.0 / display_hz
        physics_accumulator = 0.0
        frame_count = 0
        with EmbeddedCameraViewer(
            self.model,
            self.data,
            show_fixed_cameras=show_camera_panel,
        ) as viewer:
            # 窗口和OpenGL上下文初始化完成后再开始计时，避免启动耗时污染FPS。
            run_start = time.perf_counter()
            last_frame_time = run_start
            while viewer.is_running():
                frame_start = time.perf_counter()
                if max_seconds is not None and frame_start - run_start >= max_seconds:
                    break

                # 使用时间累加器把用户设置的显示节奏转换为500 Hz物理步数。
                # 当显示周期不能被0.002整除时，累加器会在相邻步数间自动
                # 调节，确保长时间平均仿真速度仍与真实时间一致。
                physics_accumulator += frame_start - last_frame_time
                last_frame_time = frame_start
                physics_steps = int(physics_accumulator / self.model.opt.timestep)
                if physics_steps > 0:
                    self.step(physics_steps)
                    physics_accumulator -= physics_steps * self.model.opt.timestep

                viewer.render()
                frame_count += 1

                remaining = frame_period - (time.perf_counter() - frame_start)
                if remaining > 0:
                    time.sleep(remaining)

        elapsed = time.perf_counter() - run_start
        return {
            "elapsed_seconds": elapsed,
            "frames": float(frame_count),
            "average_fps": frame_count / elapsed if elapsed > 0 else 0.0,
        }

    def close(self) -> None:
        """释放Headless路径使用的离屏渲染器。"""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
