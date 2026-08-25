"""独立杯子双放置区场景的回归测试。"""

from __future__ import annotations

import ast
import hashlib
import unittest
from pathlib import Path

import mujoco
import numpy as np

from sim.environment import CleanTabletopEnv
from sim.mug_environment import (
    MUG_BODY_NAME,
    MUG_FOOTPRINT_RADIUS,
    MUG_INITIAL_PAD_CLEARANCE,
    MUG_PAD_BODY_NAMES,
    MUG_PAD_GEOM_NAMES,
    MUG_SAMPLE_X_RANGE,
    MUG_SAMPLE_Y_RANGE,
    PAD_HALF_SIZE,
    MugTabletopEnv,
)


class MugTabletopEnvTest(unittest.TestCase):
    """验证杯子模型、随机重置、任务判定和相机输出。"""

    @classmethod
    def setUpClass(cls) -> None:
        """为杯子场景测试创建共享Headless环境。"""
        cls.env = MugTabletopEnv()

    @classmethod
    def tearDownClass(cls) -> None:
        """全部测试结束后释放渲染资源。"""
        cls.env.close()

    def setUp(self) -> None:
        """每个测试前恢复seed 0的稳定杯子场景。"""
        self.env.reset(scene_seed=0)

    def _place_mug_on_pad(self, pad_name: str) -> None:
        """把杯子以直立姿态放到指定区域并等待物理稳定。

        Args:
            pad_name: 蓝色或黄色区域body名称。
        """
        mug_body_id = self.env.model.body(MUG_BODY_NAME).id
        joint_id = int(self.env.model.body_jntadr[mug_body_id])
        qpos_address = int(self.env.model.jnt_qposadr[joint_id])
        pad_position = self.env.task_layout()[pad_name]["position"]
        self.env.data.qpos[qpos_address:qpos_address + 7] = [
            float(pad_position[0]),
            float(pad_position[1]),
            0.86,
            1.0,
            0.0,
            0.0,
            0.0,
        ]
        self.env.data.qvel[:] = 0.0
        mujoco.mj_forward(self.env.model, self.env.data)
        self.env.step(250)
        self.env.data.time = 0.0
        mujoco.mj_forward(self.env.model, self.env.data)

    def test_model_dimensions_objects_and_cameras(self) -> None:
        """新场景应只增加一个free杯子并保留原四路模型相机。"""
        self.assertEqual((self.env.model.nq, self.env.model.nv, self.env.model.nu), (21, 20, 7))
        self.assertGreaterEqual(self.env.model.body(MUG_BODY_NAME).id, 0)
        for body_name in MUG_PAD_BODY_NAMES:
            self.assertGreaterEqual(self.env.model.body(body_name).id, 0)
        for old_name in ("task_red_cube", "task_green_cube"):
            self.assertEqual(
                mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_BODY, old_name),
                -1,
            )
        camera_names = {self.env.model.camera(index).name for index in range(self.env.model.ncam)}
        self.assertEqual(camera_names, {"agentview", "topview", "sideview", "d435i_rgb"})

    def test_mug_contains_one_visual_and_thirty_two_collision_meshes(self) -> None:
        """杯子应保留ACT模型的一套视觉网格和32套碰撞网格。"""
        mug_body_id = self.env.model.body(MUG_BODY_NAME).id
        mug_root_id = int(self.env.model.body_rootid[mug_body_id])
        mesh_names = []
        for geom_id in range(self.env.model.ngeom):
            body_id = int(self.env.model.geom_bodyid[geom_id])
            if int(self.env.model.body_rootid[body_id]) != mug_root_id:
                continue
            mesh_id = int(self.env.model.geom_dataid[geom_id])
            if mesh_id >= 0 and self.env.model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH:
                mesh_names.append(self.env.model.mesh(mesh_id).name)
        self.assertEqual(sum(name.endswith("_vis") for name in mesh_names), 1)
        self.assertEqual(sum("_coll" in name for name in mesh_names), 32)

    def test_source_mug_files_are_copied_byte_for_byte(self) -> None:
        """除新增兼容XML外，ACT杯子目录中的文件应逐字节一致。"""
        project_root = Path(__file__).resolve().parents[1]
        source_root = Path(r"F:\桌面\code_learn\mujoco-act-robotics\mode\mug_5")
        target_root = project_root / "assets" / "mujoco" / "mug_5"
        for source_path in sorted(path for path in source_root.rglob("*") if path.is_file()):
            relative_path = source_path.relative_to(source_root)
            target_path = target_root / relative_path
            self.assertTrue(target_path.is_file(), f"缺少杯子资源: {relative_path}")
            self.assertEqual(
                hashlib.sha256(source_path.read_bytes()).digest(),
                hashlib.sha256(target_path.read_bytes()).digest(),
                f"杯子源资源发生变化: {relative_path}",
            )

    def test_seeded_reset_is_reproducible_and_clear_of_pads(self) -> None:
        """同seed应复现，不同seed应变化且杯子初始不覆盖区域。"""
        first = self.env.reset(scene_seed=7)
        repeated = self.env.reset(scene_seed=7)
        different = self.env.reset(scene_seed=8)
        np.testing.assert_allclose(first.mug_initial_pose, repeated.mug_initial_pose, atol=1e-12)
        self.assertFalse(np.allclose(first.mug_initial_pose, different.mug_initial_pose))
        for snapshot in (first, repeated, different):
            x, y = snapshot.mug_initial_pose[:2]
            self.assertTrue(MUG_SAMPLE_X_RANGE[0] - 0.01 <= x <= MUG_SAMPLE_X_RANGE[1] + 0.01)
            self.assertTrue(MUG_SAMPLE_Y_RANGE[0] - 0.01 <= y <= MUG_SAMPLE_Y_RANGE[1] + 0.01)
            expanded = np.asarray(PAD_HALF_SIZE[:2]) + (
                MUG_FOOTPRINT_RADIUS + MUG_INITIAL_PAD_CLEARANCE
            )
            for pad_position in snapshot.pad_positions:
                self.assertFalse(bool(np.all(np.abs(snapshot.mug_initial_pose[:2] - pad_position[:2]) <= expanded)))

    def test_reset_returns_a_stable_upright_mug(self) -> None:
        """250步稳定化后杯子应直立落桌且速度接近零。"""
        self.env.reset(scene_seed=11)
        mug = self.env.task_layout()[MUG_BODY_NAME]
        self.assertAlmostEqual(float(mug["position"][2]), 0.84465, delta=2e-4)
        self.assertLess(float(mug["linear_speed"]), 1e-8)
        self.assertLess(float(mug["angular_speed"]), 1e-8)
        self.assertGreater(
            float(mug["top_site_position"][2] - mug["bottom_site_position"][2]),
            0.07,
        )
        self.assertAlmostEqual(float(mug["bottom_site_position"][2]), 0.8, delta=0.015)

    def test_pad_properties_are_unchanged(self) -> None:
        """新场景的蓝黄区域应与积木场景保持同一参数。"""
        layout = self.env.task_layout()
        expected_positions = ([0.55, -0.22, 0.8005], [0.55, 0.22, 0.8005])
        for body_name, geom_name, expected_position in zip(
            MUG_PAD_BODY_NAMES,
            MUG_PAD_GEOM_NAMES,
            expected_positions,
            strict=True,
        ):
            self.assertGreaterEqual(self.env.model.geom(geom_name).id, 0)
            np.testing.assert_allclose(layout[body_name]["position"], expected_position, atol=1e-12)
            np.testing.assert_allclose(layout[body_name]["geom_half_size"], PAD_HALF_SIZE, atol=1e-12)
            self.assertEqual(layout[body_name]["mass"], 0.0)
            self.assertEqual(layout[body_name]["contype"], 0)
            self.assertEqual(layout[body_name]["conaffinity"], 0)

    def test_blue_and_yellow_tasks_require_stability_and_release(self) -> None:
        """杯子在两个目标区稳定且松爪后都应通过严格成功判定。"""
        for task_id, pad_name in (
            ("mug_on_blue", "task_blue_pad"),
            ("mug_on_yellow", "task_yellow_pad"),
        ):
            self.env.reset(scene_seed=2)
            self._place_mug_on_pad(pad_name)
            pending = self.env.evaluate_task(task_id)
            self.assertFalse(pending.success)
            self.env.data.time += 0.5
            success = self.env.evaluate_task(task_id)
            self.assertTrue(success.success)
            self.assertEqual(success.failure_mode, "success")
            self.assertTrue(success.metrics["target_inside"])
            self.assertAlmostEqual(
                float(success.metrics["bottom_z"]),
                float(self.env.task_layout()[MUG_BODY_NAME]["bottom_site_position"][2]),
            )

    def test_wrong_pad_is_classified(self) -> None:
        """杯子稳定放到另一颜色区域时应分类为wrong_pad。"""
        self._place_mug_on_pad("task_yellow_pad")
        result = self.env.evaluate_task("mug_on_blue")
        self.assertFalse(result.success)
        self.assertEqual(result.failure_mode, "wrong_pad")

    def test_robot_table_and_camera_poses_match_block_scene(self) -> None:
        """新增杯子不得改变机器人、桌面和四路模型相机世界位姿。"""
        with CleanTabletopEnv() as block_env:
            block_layout = block_env.spatial_layout()
            mug_layout = self.env.spatial_layout()
            for key in (
                "base_position",
                "base_rotation",
                "table_body_position",
                "table_body_rotation",
                "table_geom_position",
                "table_half_size",
            ):
                np.testing.assert_allclose(mug_layout[key], block_layout[key], atol=1e-9)
            for camera_name in ("agentview", "topview", "sideview", "d435i_rgb"):
                mug_camera = self.env.model.camera(camera_name).id
                block_camera = block_env.model.camera(camera_name).id
                np.testing.assert_allclose(
                    self.env.data.cam_xpos[mug_camera],
                    block_env.data.cam_xpos[block_camera],
                    atol=1e-9,
                )
                np.testing.assert_allclose(
                    self.env.data.cam_xmat[mug_camera],
                    block_env.data.cam_xmat[block_camera],
                    atol=1e-9,
                )
                self.assertAlmostEqual(
                    float(self.env.model.cam_fovy[mug_camera]),
                    float(block_env.model.cam_fovy[block_camera]),
                    places=9,
                )

    def test_headless_cameras_are_non_empty_rgb(self) -> None:
        """新场景应能推进并渲染三路非空RGB图像。"""
        self.env.step(10)
        for image in self.env.capture_cameras().values():
            self.assertEqual(image.shape, (256, 256, 3))
            self.assertEqual(image.dtype, np.uint8)
            self.assertGreater(float(image.mean()), 0.0)

    def test_green_white_variant_changes_only_visual_appearance(self) -> None:
        """绿白纹理应改变相机图像，但保持场景物理参数和原色恢复一致。"""
        seed = 2291
        self.env.reset(seed)
        original_images = self.env.capture_training_images()
        original_state = self.env.get_state()
        with MugTabletopEnv(appearance_variant="original") as restored_env:
            restored_snapshot = restored_env.reset(seed)
            restored_images = restored_env.capture_training_images()
            np.testing.assert_allclose(
                restored_images["agent"],
                original_images["agent"],
                atol=1,
            )
            np.testing.assert_allclose(
                restored_images["wrist"],
                original_images["wrist"],
                atol=1,
            )
        with MugTabletopEnv(appearance_variant="green_white") as green_env:
            green_snapshot = green_env.reset(seed)
            green_images = green_env.capture_training_images()
            np.testing.assert_allclose(green_snapshot.mug_initial_pose, restored_snapshot.mug_initial_pose)
            np.testing.assert_allclose(green_env.get_state(), original_state)
            for attribute in (
                "body_mass",
                "body_inertia",
                "geom_friction",
                "geom_size",
                "mesh_scale",
                "actuator_ctrlrange",
            ):
                np.testing.assert_allclose(
                    getattr(green_env.model, attribute),
                    getattr(restored_env.model, attribute),
                )
            self.assertGreater(
                int(np.count_nonzero(green_images["agent"] != original_images["agent"])),
                100,
            )
            self.assertGreater(
                int(np.count_nonzero(green_images["wrist"] != original_images["wrist"])),
                100,
            )

    def test_unknown_appearance_variant_is_rejected(self) -> None:
        """未知杯子外观变体必须明确报错。"""
        with self.assertRaisesRegex(ValueError, "未知杯子外观变体"):
            MugTabletopEnv(appearance_variant="blue")


class MugDomainRandomizationTest(unittest.TestCase):
    """验证环境级域随机化（光照）的确定性、渲染生效与正交性。"""

    def test_set_domain_is_deterministic(self) -> None:
        """同一 domain_seed 必须严格复现同一组光照参数。"""
        with MugTabletopEnv() as env:
            first = env.set_domain(42)
            second = env.set_domain(42)
            self.assertEqual(first, second)
            self.assertEqual(set(first), {"domain_seed", "light_A_scale", "light_B_azimuth_deg", "light_C_scale"})
            self.assertIn(first["domain_seed"], (42,))
            for key in ("light_A_scale", "light_C_scale"):
                self.assertTrue(0.4 <= first[key] <= 1.6)
            self.assertTrue(-36.0 <= first["light_B_azimuth_deg"] <= 36.0)

    def test_set_domain_changes_render(self) -> None:
        """随机化光照必须实际改变 agent 相机渲染像素。"""
        with MugTabletopEnv() as env:
            env.reset(scene_seed=7)
            default_image = env.capture_camera("agentview")
            env.set_domain(2024)
            env.reset(scene_seed=7)
            randomized_image = env.capture_camera("agentview")
            changed = int(np.count_nonzero(randomized_image != default_image))
            self.assertGreater(changed, 1000)

    def test_domain_is_orthogonal_to_scene_seed(self) -> None:
        """光照参数只由 domain_seed 决定，与 reset 的 scene_seed 无关。"""
        with MugTabletopEnv() as env:
            env.set_domain(42)
            env.reset(scene_seed=1)
            first = env.domain_summary()
            env.set_domain(42)
            env.reset(scene_seed=2)
            second = env.domain_summary()
            self.assertEqual(first, second)

    def test_reset_remains_compatible_after_domain(self) -> None:
        """set_domain 之后旧 reset(scene_seed) 契约保持成立。"""
        with MugTabletopEnv() as env:
            env.set_domain(99)
            snapshot = env.reset(scene_seed=0)
            self.assertEqual(snapshot.scene_seed, 0)
            self.assertTrue(np.isfinite(snapshot.mug_initial_pose).all())
            self.assertEqual(snapshot.pad_positions.shape, (2, 3))


class MugDocumentationTest(unittest.TestCase):
    """检查新增杯子Python接口的中文文档字符串。"""

    def test_new_definitions_have_docstrings(self) -> None:
        """新增类、函数和方法都应包含文档字符串。"""
        project_root = Path(__file__).resolve().parents[1]
        missing = []
        for source_path in (
            project_root / "sim" / "mug_environment.py",
            project_root / "view_mug_scene.py",
        ):
            tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
            for node in ast.walk(tree):
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    if ast.get_docstring(node) is None:
                        missing.append(f"{source_path.name}:{node.lineno}:{node.name}")
        self.assertEqual(missing, [], f"以下新增定义缺少文档字符串: {missing}")


if __name__ == "__main__":
    unittest.main()
