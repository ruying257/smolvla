"""ACT双积木桌面场景的回归测试。"""

from __future__ import annotations

import ast
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from sim.environment import (
    ACT_INITIAL_ARM_QPOS_DEG,
    ARM_JOINT_NAMES,
    CUBE_HALF_SIZE,
    CUBE_MIN_CENTER_DISTANCE,
    CUBE_SAMPLE_X_RANGE,
    CUBE_SAMPLE_Y_RANGE,
    PAD_HALF_SIZE,
    TASK_CUBE_BODY_NAMES,
    TASK_CUBE_GEOM_NAMES,
    TASK_INITIAL_BODY_POSITIONS,
    TASK_PAD_BODY_NAMES,
    CleanTabletopEnv,
)


class CleanTabletopEnvTest(unittest.TestCase):
    """验证模型规模、任务布局、初始状态和相机输出。"""

    @classmethod
    def setUpClass(cls) -> None:
        """为全部测试创建一个共享的 Headless 环境。"""
        cls.env = CleanTabletopEnv()

    @classmethod
    def tearDownClass(cls) -> None:
        """测试结束后释放共享渲染资源。"""
        cls.env.close()

    def setUp(self) -> None:
        """在每个测试前恢复种子0对应的机器人和积木状态。"""
        self.env.reset(scene_seed=0)

    def test_model_dimensions_and_cameras(self) -> None:
        """模型自由度、控制维度和相机集合应与 ACT 场景一致。"""
        self.assertEqual(self.env.model.nq, 28)
        self.assertEqual(self.env.model.nv, 26)
        self.assertEqual(self.env.model.nu, 7)
        camera_names = {self.env.model.camera(index).name for index in range(self.env.model.ncam)}
        self.assertEqual(camera_names, {"agentview", "topview", "sideview", "d435i_rgb"})

    def test_act_initial_joint_angles(self) -> None:
        """六个机械臂关节应保持 ACT 使用的初始角度。"""
        actual = []
        for joint_name in ARM_JOINT_NAMES:
            joint_id = self.env.model.joint(joint_name).id
            actual.append(self.env.data.qpos[self.env.model.jnt_qposadr[joint_id]])
        np.testing.assert_allclose(actual, np.deg2rad(ACT_INITIAL_ARM_QPOS_DEG), atol=1e-12)

    def test_robot_table_spatial_layout(self) -> None:
        """机械臂基座与桌面的世界和相对位置必须保持 ACT 数值。"""
        layout = self.env.spatial_layout()
        np.testing.assert_allclose(layout["base_position"], [1.0, 0.0, 0.8], atol=1e-12)
        np.testing.assert_allclose(layout["table_body_position"], [0.0, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(layout["table_geom_position"], [0.5, 0.0, 0.4], atol=1e-12)
        np.testing.assert_allclose(layout["table_half_size"], [1.0, 0.7, 0.4], atol=1e-12)
        np.testing.assert_allclose(layout["base_from_table_body"], [1.0, 0.0, 0.8], atol=1e-12)
        np.testing.assert_allclose(layout["base_from_table_geom"], [0.5, 0.0, 0.4], atol=1e-12)
        self.assertAlmostEqual(float(layout["table_top_z"]), 0.8, places=12)

    def test_task_objects_have_locked_layout_and_properties(self) -> None:
        """积木和放置区域应具有方案锁定的名称、位姿及物理属性。"""
        layout = self.env.task_layout()
        expected_positions = dict(TASK_INITIAL_BODY_POSITIONS)
        self.assertEqual(
            set(layout),
            {*TASK_CUBE_BODY_NAMES, *TASK_PAD_BODY_NAMES},
        )

        for body_name in TASK_CUBE_BODY_NAMES:
            item = layout[body_name]
            self.assertTrue(CUBE_SAMPLE_X_RANGE[0] <= item["position"][0] <= CUBE_SAMPLE_X_RANGE[1])
            self.assertTrue(CUBE_SAMPLE_Y_RANGE[0] <= item["position"][1] <= CUBE_SAMPLE_Y_RANGE[1])
            self.assertAlmostEqual(float(item["position"][2]), 0.825, places=12)
            np.testing.assert_allclose(item["geom_half_size"], CUBE_HALF_SIZE, atol=1e-12)
            np.testing.assert_allclose(item["friction"], [1.0, 0.005, 0.0001], atol=1e-12)
            self.assertAlmostEqual(float(item["mass"]), 0.05, places=12)

        np.testing.assert_allclose(layout["task_red_cube"]["rgba"], [0.85, 0.1, 0.1, 1.0], atol=1e-7)
        np.testing.assert_allclose(layout["task_green_cube"]["rgba"], [0.1, 0.75, 0.2, 1.0], atol=1e-7)

        for body_name in TASK_PAD_BODY_NAMES:
            item = layout[body_name]
            np.testing.assert_allclose(item["position"], expected_positions[body_name], atol=1e-12)
            np.testing.assert_allclose(item["geom_half_size"], PAD_HALF_SIZE, atol=1e-12)
            self.assertEqual(item["contype"], 0)
            self.assertEqual(item["conaffinity"], 0)
            self.assertEqual(item["mass"], 0.0)

        np.testing.assert_allclose(layout["task_blue_pad"]["rgba"], [0.1, 0.3, 0.9, 0.65], atol=1e-7)
        np.testing.assert_allclose(layout["task_yellow_pad"]["rgba"], [0.95, 0.8, 0.1, 0.65], atol=1e-7)

    def test_cube_reset_is_seeded_and_non_overlapping(self) -> None:
        """同seed应复现布局，不同seed应变化且两个积木保持12厘米间距。"""
        expected_positions = dict(TASK_INITIAL_BODY_POSITIONS)
        first = self.env.reset(scene_seed=7)
        repeated = self.env.reset(scene_seed=7)
        different = self.env.reset(scene_seed=8)
        np.testing.assert_allclose(first.cube_initial_poses, repeated.cube_initial_poses, atol=1e-12)
        self.assertFalse(np.allclose(first.cube_initial_poses, different.cube_initial_poses))
        for snapshot in (first, repeated, different):
            distance = np.linalg.norm(
                snapshot.cube_initial_poses[0, :2] - snapshot.cube_initial_poses[1, :2]
            )
            self.assertGreaterEqual(float(distance), CUBE_MIN_CENTER_DISTANCE)
            np.testing.assert_allclose(
                snapshot.pad_positions,
                [expected_positions[body_name] for body_name in TASK_PAD_BODY_NAMES],
                atol=1e-12,
            )

    def test_cubes_remain_stable_after_one_thousand_steps(self) -> None:
        """两个积木应稳定落在桌面上且不产生异常水平漂移。"""
        snapshot = self.env.reset(scene_seed=11)
        self.env.step(1000)
        layout = self.env.task_layout()
        for index, (body_name, joint_name) in enumerate(zip(
            TASK_CUBE_BODY_NAMES,
            ("task_red_cube_free_joint", "task_green_cube_free_joint"),
            strict=True,
        )):
            position = layout[body_name]["position"]
            np.testing.assert_allclose(position[:2], snapshot.cube_initial_poses[index, :2], atol=1e-8)
            self.assertAlmostEqual(float(position[2]), 0.825, delta=2e-4)
            joint_id = mujoco.mj_name2id(
                self.env.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_name,
            )
            dof_address = int(self.env.model.jnt_dofadr[joint_id])
            self.assertLess(float(np.linalg.norm(self.env.data.qvel[dof_address:dof_address + 6])), 1e-8)

    def test_strict_success_requires_correct_cube_stability_and_release(self) -> None:
        """正确积木稳定落入目标区且夹爪释放后才返回严格成功。"""
        self.env.reset(scene_seed=3)
        red_joint_id = self.env.model.joint("task_red_cube_free_joint").id
        red_qpos_address = int(self.env.model.jnt_qposadr[red_joint_id])
        blue_position = self.env.task_layout()["task_blue_pad"]["position"]
        self.env.data.qpos[red_qpos_address:red_qpos_address + 3] = [
            blue_position[0],
            blue_position[1],
            0.825,
        ]
        self.env.data.qvel[:] = 0.0
        self.env.data.ctrl[6] = 0.0
        mujoco.mj_forward(self.env.model, self.env.data)

        pending = self.env.evaluate_task("red_on_blue")
        self.assertFalse(pending.success)
        self.env.data.time += 0.5
        success = self.env.evaluate_task("red_on_blue")
        self.assertTrue(success.success)
        self.assertEqual(success.failure_mode, "success")

    def test_wrong_cube_on_target_pad_is_classified(self) -> None:
        """即使目标积木正确，非目标积木也会阻止成功并分类为抓错。"""
        self.env.reset(scene_seed=4)
        red_joint_id = self.env.model.joint("task_red_cube_free_joint").id
        green_joint_id = self.env.model.joint("task_green_cube_free_joint").id
        red_qpos_address = int(self.env.model.jnt_qposadr[red_joint_id])
        green_qpos_address = int(self.env.model.jnt_qposadr[green_joint_id])
        blue_position = self.env.task_layout()["task_blue_pad"]["position"]
        self.env.data.qpos[red_qpos_address:red_qpos_address + 3] = [
            blue_position[0] - 0.02,
            blue_position[1],
            0.825,
        ]
        self.env.data.qpos[green_qpos_address:green_qpos_address + 3] = [
            blue_position[0] + 0.02,
            blue_position[1],
            0.825,
        ]
        self.env.data.qvel[:] = 0.0
        mujoco.mj_forward(self.env.model, self.env.data)
        self.env.evaluate_task("red_on_blue")
        self.env.data.time += 0.5
        result = self.env.evaluate_task("red_on_blue")
        self.assertFalse(result.success)
        self.assertEqual(result.failure_mode, "wrong_cube")

    def test_scene_xml_uses_demo_visuals_without_cup_or_plate(self) -> None:
        """主XML应使用ACT演示视觉配置，并彻底移除杯盘引用。"""
        scene_path = self.env.scene_path
        scene_text = scene_path.read_text(encoding="utf-8")
        self.assertNotIn("plate_11", scene_text)
        self.assertNotIn("mug_5", scene_text)
        self.assertNotIn("floor_mujoco_style.xml", scene_text)
        self.assertNotIn("floor_isaac_style.xml", scene_text)

        root = ET.fromstring(scene_text)
        self.assertEqual(root.find("statistic").attrib, {"center": "0.4 0 0.4", "extent": "1"})
        self.assertEqual(root.find("visual/global").attrib, {"azimuth": "120", "elevation": "-20"})
        self.assertIsNotNone(root.find("worldbody/geom[@name='floor']"))
        for geom_name in ("world_axis_x", "world_axis_y", "world_axis_z"):
            self.assertIsNotNone(root.find(f"worldbody/geom[@name='{geom_name}']"))
        for geom_name in TASK_CUBE_GEOM_NAMES:
            self.assertIsNotNone(root.find(f"worldbody/body/geom[@name='{geom_name}']"))

    def test_headless_step_and_camera_render(self) -> None:
        """环境应能推进十步并渲染三路非空 RGB 图像。"""
        self.env.step(10)
        for image in self.env.capture_cameras().values():
            self.assertEqual(image.shape, (256, 256, 3))
            self.assertEqual(image.dtype, np.uint8)
            self.assertGreater(float(image.mean()), 0.0)


class DocumentationTest(unittest.TestCase):
    """检查本任务新增 Python API 的文档字符串完整性。"""

    def test_python_definitions_have_docstrings(self) -> None:
        """所有类、函数和方法都必须包含中文Google风格文档字符串。"""
        project_root = Path(__file__).resolve().parents[1]
        source_paths = [
            project_root / "view_scene.py",
            project_root / "sim" / "environment.py",
            project_root / "sim" / "mujoco_viewer.py",
            project_root / "scripts" / "verify_act_layout.py",
            project_root / "scripts" / "generate_asset_manifest.py",
        ]
        source_paths.extend(sorted((project_root / "collector").glob("*.py")))
        source_paths.extend(sorted((project_root / "cloud").glob("*.py")))
        missing = []
        for source_path in source_paths:
            tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
            for node in ast.walk(tree):
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    if ast.get_docstring(node) is None:
                        missing.append(f"{source_path.name}:{node.lineno}:{node.name}")
        self.assertEqual(missing, [], f"以下定义缺少文档字符串: {missing}")


if __name__ == "__main__":
    unittest.main()
