"""UR10e键盘末端增量控制与阻尼最小二乘IK。"""

from __future__ import annotations

from dataclasses import dataclass

import glfw
import mujoco
import numpy as np
from numpy.typing import NDArray

from sim.environment import ARM_JOINT_NAMES, CleanTabletopEnv
from sim.mujoco_viewer import EmbeddedCameraViewer


IK_NOT_CONVERGED_MESSAGE = "IK did not converge; holding previous joint target"


@dataclass(frozen=True)
class TeleopDelta:
    """一次键盘采样产生的末端增量和夹爪状态。"""

    translation: NDArray[np.float64]
    rotation_rpy: NDArray[np.float64]
    gripper: float
    meaningful: bool


def read_teleop_delta(viewer: EmbeddedCameraViewer, gripper: float) -> TeleopDelta:
    """按ACT键位读取一次末端位姿增量。

    Args:
        viewer: 提供持续按键和单次按键事件的Viewer。
        gripper: 进入本次采样前的0/1夹爪指令。

    Returns:
        平移、旋转、夹爪状态和是否发生有效操作。
    """
    translation = np.zeros(3, dtype=np.float64)
    rotation = np.zeros(3, dtype=np.float64)
    key_vectors = (
        (glfw.KEY_S, (0.007, 0.0, 0.0)),
        (glfw.KEY_W, (-0.007, 0.0, 0.0)),
        (glfw.KEY_A, (0.0, -0.007, 0.0)),
        (glfw.KEY_D, (0.0, 0.007, 0.0)),
        (glfw.KEY_R, (0.0, 0.0, 0.007)),
        (glfw.KEY_F, (0.0, 0.0, -0.007)),
    )
    for key, vector in key_vectors:
        if viewer.is_key_down(key):
            translation += np.asarray(vector)
    rotation_keys = (
        (glfw.KEY_DOWN, (0.03, 0.0, 0.0)),
        (glfw.KEY_UP, (-0.03, 0.0, 0.0)),
        (glfw.KEY_LEFT, (0.0, 0.03, 0.0)),
        (glfw.KEY_RIGHT, (0.0, -0.03, 0.0)),
        (glfw.KEY_Q, (0.0, 0.0, 0.03)),
        (glfw.KEY_E, (0.0, 0.0, -0.03)),
    )
    for key, vector in rotation_keys:
        if viewer.is_key_down(key):
            rotation += np.asarray(vector)
    toggled = viewer.consume_key_press(glfw.KEY_SPACE)
    next_gripper = 1.0 - gripper if toggled else gripper
    meaningful = bool(np.any(translation) or np.any(rotation) or toggled)
    return TeleopDelta(translation, rotation, next_gripper, meaningful)


def _rotation_from_rpy(rpy: NDArray[np.float64]) -> NDArray[np.float64]:
    """把XYZ欧拉角增量转换为旋转矩阵。

    Args:
        rpy: 三维滚转、俯仰和偏航角。

    Returns:
        三乘三旋转矩阵。
    """
    roll, pitch, yaw = rpy
    cx, sx = np.cos(roll), np.sin(roll)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cz, sz = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
            [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
            [-sy, cy * sx, cy * cx],
        ],
        dtype=np.float64,
    )


class DifferentialIKController:
    """使用MuJoCo Jacobian把末端增量转换为绝对关节目标。"""

    def __init__(self, env: CleanTabletopEnv, body_name: str = "wrist_3_link") -> None:
        """缓存关节地址并从当前末端位姿初始化控制目标。

        Args:
            env: 当前真实MuJoCo环境。
            body_name: IK目标body名称。
        """
        self.env = env
        self.body_id = env.model.body(body_name).id
        self.qpos_addresses = np.array(
            [env.model.jnt_qposadr[env.model.joint(name).id] for name in ARM_JOINT_NAMES],
            dtype=np.int32,
        )
        self.dof_addresses = np.array(
            [env.model.jnt_dofadr[env.model.joint(name).id] for name in ARM_JOINT_NAMES],
            dtype=np.int32,
        )
        self.scratch = mujoco.MjData(env.model)
        self.gripper = 0.0
        self.target_position = np.zeros(3)
        self.target_rotation = np.eye(3)
        self.last_action = np.zeros(7, dtype=np.float32)
        self.reset()

    def reset(self) -> None:
        """从环境当前状态重置末端目标和最后动作。"""
        mujoco.mj_forward(self.env.model, self.env.data)
        self.target_position = self.env.data.xpos[self.body_id].copy()
        self.target_rotation = self.env.data.xmat[self.body_id].reshape(3, 3).copy()
        state = self.env.get_state()
        self.gripper = float(state[6])
        self.last_action = state.copy()

    def command(self, delta: TeleopDelta) -> tuple[NDArray[np.float32], str]:
        """求解一个末端增量对应的七维绝对动作。

        Args:
            delta: 本次键盘末端增量。

        Returns:
            ``(action, error)``；失败时返回上一动作和非空错误文本。
        """
        self.gripper = delta.gripper
        if not delta.meaningful or (not np.any(delta.translation) and not np.any(delta.rotation_rpy)):
            action = self.last_action.copy()
            action[6] = self.gripper
            self.last_action = action
            return action, ""
        candidate_position = self.target_position + delta.translation
        candidate_rotation = self.target_rotation @ _rotation_from_rpy(delta.rotation_rpy)
        solved = self._solve(candidate_position, candidate_rotation)
        if solved is None:
            action = self.last_action.copy()
            action[6] = self.gripper
            self.last_action = action
            return action, IK_NOT_CONVERGED_MESSAGE
        self.target_position = candidate_position
        self.target_rotation = candidate_rotation
        self.last_action = np.asarray([*solved, self.gripper], dtype=np.float32)
        return self.last_action.copy(), ""

    def _solve(
        self,
        target_position: NDArray[np.float64],
        target_rotation: NDArray[np.float64],
    ) -> NDArray[np.float64] | None:
        """执行阻尼最小二乘迭代IK。

        Args:
            target_position: 目标世界坐标。
            target_rotation: 目标世界旋转矩阵。

        Returns:
            收敛后的六关节角；失败时返回 ``None``。
        """
        self.scratch.qpos[:] = self.env.data.qpos
        self.scratch.qvel[:] = 0.0
        joint_ranges = self.env.model.jnt_range[
            [self.env.model.joint(name).id for name in ARM_JOINT_NAMES]
        ]
        for _ in range(60):
            mujoco.mj_forward(self.env.model, self.scratch)
            current_position = self.scratch.xpos[self.body_id]
            current_rotation = self.scratch.xmat[self.body_id].reshape(3, 3)
            position_error = target_position - current_position
            rotation_error = 0.5 * (
                np.cross(current_rotation[:, 0], target_rotation[:, 0])
                + np.cross(current_rotation[:, 1], target_rotation[:, 1])
                + np.cross(current_rotation[:, 2], target_rotation[:, 2])
            )
            error = np.concatenate([position_error, rotation_error])
            if np.linalg.norm(position_error) < 1e-3 and np.linalg.norm(rotation_error) < 2e-2:
                return self.scratch.qpos[self.qpos_addresses].copy()
            jac_position = np.zeros((3, self.env.model.nv))
            jac_rotation = np.zeros((3, self.env.model.nv))
            mujoco.mj_jacBody(
                self.env.model,
                self.scratch,
                jac_position,
                jac_rotation,
                self.body_id,
            )
            jacobian = np.vstack([jac_position, jac_rotation])[:, self.dof_addresses]
            damping = 1e-4
            delta_q = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + damping * np.eye(6),
                error,
            )
            next_q = self.scratch.qpos[self.qpos_addresses] + np.clip(delta_q, -0.15, 0.15)
            next_q = np.clip(next_q, joint_ranges[:, 0], joint_ranges[:, 1])
            self.scratch.qpos[self.qpos_addresses] = next_q
        return None
