"""P2采集控制、状态机和数据契约回归测试。"""

from __future__ import annotations

import unittest
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np

from collector.collect import (
    NOTICE_CANCELLED,
    NOTICE_DISCARDED,
    NOTICE_SUCCESS,
    NOTICE_TIMEOUT,
    _status_text,
)
from collector.control import (
    IK_NOT_CONVERGED_MESSAGE,
    DifferentialIKController,
    TeleopDelta,
)
from collector.dataset_io import (
    DATASET_FPS,
    DATASET_VERSION,
    LeRobotEpisodeWriter,
    _accessible_mkdtemp,
    contract_mismatches,
    concatenate_video_files_utf8,
    dataset_features,
    expected_contract,
    find_ffmpeg,
)
from collector.state_machine import CollectionPhase, CollectionStateMachine
from collector.task_spec import TASKS, choose_balanced_template
from sim.environment import CleanTabletopEnv


class CollectionStateMachineTest(unittest.TestCase):
    """验证首次操作启动、成功暂停和人工确认边界。"""

    def test_first_meaningful_action_starts_recording(self) -> None:
        """静止帧不启动录制，首次有效操作应成为第一帧。"""
        machine = CollectionStateMachine()
        self.assertFalse(machine.observe_action(False))
        self.assertTrue(machine.observe_action(True))
        self.assertEqual(machine.phase, CollectionPhase.RECORDING)
        self.assertEqual(machine.frame_count, 1)

    def test_success_requires_confirmation_before_reset(self) -> None:
        """严格成功后应等待确认，确认后才回到空闲状态。"""
        machine = CollectionStateMachine()
        machine.observe_action(True)
        self.assertTrue(machine.observe_success(True))
        self.assertEqual(machine.phase, CollectionPhase.PENDING_CONFIRMATION)
        self.assertTrue(machine.confirm())
        self.assertEqual(machine.phase, CollectionPhase.IDLE)
        self.assertEqual(machine.frame_count, 0)

    def test_discard_clears_recorded_frames(self) -> None:
        """取消或人工丢弃应清空状态机帧计数。"""
        machine = CollectionStateMachine()
        machine.observe_action(True)
        machine.observe_action(False)
        self.assertTrue(machine.discard())
        self.assertEqual(machine.phase, CollectionPhase.IDLE)
        self.assertEqual(machine.frame_count, 0)

    def test_viewer_status_messages_are_ascii(self) -> None:
        """MuJoCo内置overlay不支持中文，所有运行时提示必须为ASCII。"""
        notices = (
            NOTICE_CANCELLED,
            NOTICE_DISCARDED,
            NOTICE_SUCCESS,
            NOTICE_TIMEOUT,
            IK_NOT_CONVERGED_MESSAGE,
        )
        for notice in notices:
            status = _status_text(
                CollectionPhase.RECORDING,
                "Put the red cube on the blue pad.",
                scene_seed=0,
                frames=1,
                saved=0,
                target=1,
                error=notice,
            )
            self.assertTrue(status.isascii(), status)


class TaskTemplateTest(unittest.TestCase):
    """验证四类英文任务和两种训练措辞均衡规则。"""

    def test_all_four_canonical_tasks_are_locked(self) -> None:
        """四类canonical文本应与计划逐字一致。"""
        prompts = {task_id: task.prompt("canonical") for task_id, task in TASKS.items()}
        self.assertEqual(prompts["red_on_blue"], "Put the red cube on the blue pad.")
        self.assertEqual(prompts["red_on_yellow"], "Put the red cube on the yellow pad.")
        self.assertEqual(prompts["green_on_blue"], "Put the green cube on the blue pad.")
        self.assertEqual(prompts["green_on_yellow"], "Put the green cube on the yellow pad.")

    def test_template_selection_balances_saved_counts(self) -> None:
        """数量相同时选canonical，否则选择当前较少模板。"""
        self.assertEqual(choose_balanced_template("red_on_blue", {}), "canonical")
        counts = {"red_on_blue/canonical": 1, "red_on_blue/synonym": 0}
        self.assertEqual(choose_balanced_template("red_on_blue", counts), "synonym")


class DatasetContractTest(unittest.TestCase):
    """验证LeRobot feature schema和续采拒绝规则。"""

    def test_feature_schema_shapes_and_dtypes(self) -> None:
        """六个受控feature必须保持计划锁定的shape和dtype。"""
        features = dataset_features()
        self.assertEqual(DATASET_FPS, 20)
        self.assertEqual(features["observation.images.agent"]["shape"], (256, 256, 3))
        self.assertEqual(features["observation.images.wrist"]["dtype"], "video")
        self.assertEqual(features["observation.state"]["shape"], (7,))
        self.assertEqual(features["action"]["shape"], (7,))
        self.assertEqual(features["scene_seed"]["dtype"], "int64")
        self.assertEqual(features["cube_initial_poses"]["shape"], (14,))

    def test_contract_records_fixed_pads_and_randomization(self) -> None:
        """数据集级契约应记录版本、固定区域和随机化规格。"""
        contract = expected_contract()
        self.assertEqual(contract["dataset_version"], DATASET_VERSION)
        self.assertEqual(contract["fixed_pad_positions"]["blue"], [0.55, -0.22, 0.8005])
        self.assertEqual(contract["fixed_pad_positions"]["yellow"], [0.55, 0.22, 0.8005])
        self.assertEqual(contract["cube_min_center_distance"], 0.12)

    def test_ffmpeg_preflight_finds_available_encoder(self) -> None:
        """采集环境应能找到系统或Python包提供的FFmpeg。"""
        executable = find_ffmpeg()
        self.assertIsNotNone(executable)
        self.assertTrue(executable.is_file())

    def test_video_concatenation_supports_chinese_paths(self) -> None:
        """第二个episode的视频应能追加到中文目录中的已有视频。"""
        import av

        def write_video(path: Path, color: int) -> None:
            """写入两帧用于验证remux的H.264视频。"""
            path.parent.mkdir(parents=True, exist_ok=True)
            with av.open(str(path), "w") as container:
                stream = container.add_stream("h264", rate=20)
                stream.width = 32
                stream.height = 32
                stream.pix_fmt = "yuv420p"
                for _ in range(2):
                    image = np.full((32, 32, 3), color, dtype=np.uint8)
                    frame = av.VideoFrame.from_ndarray(image, format="rgb24")
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)

        temporary_directory = Path(_accessible_mkdtemp(dir=Path.cwd()))
        try:
            chinese_directory = temporary_directory / "中文路径"
            first_path = chinese_directory / "first.mp4"
            second_path = chinese_directory / "second.mp4"
            output_path = chinese_directory / "output.mp4"
            write_video(first_path, 32)
            write_video(second_path, 224)

            concatenate_video_files_utf8([first_path, second_path], output_path)

            with av.open(str(output_path)) as container:
                decoded_frames = sum(1 for _ in container.decode(video=0))
            self.assertEqual(decoded_frames, 4)
        finally:
            shutil.rmtree(temporary_directory, ignore_errors=True)

    def test_discard_removes_video_feature_png_directories(self) -> None:
        """取消episode时必须清理LeRobot未覆盖的video临时PNG目录。"""
        temporary_directory = Path(_accessible_mkdtemp(dir=Path.cwd()))
        try:
            video_directory = temporary_directory / "images" / "agent" / "episode-000000"
            video_directory.mkdir(parents=True)
            (video_directory / "frame-000000.png").write_bytes(b"stale")

            class FakeDataset:
                """提供丢弃逻辑所需的最小LeRobot数据集接口。"""

                def __init__(self) -> None:
                    self.meta = SimpleNamespace(video_keys=["observation.images.agent"])
                    self.episode_buffer = {"episode_index": 0}
                    self.cleared = False

                def _wait_image_writer(self) -> None:
                    """模拟等待异步图像写入完成。"""

                def _get_image_file_dir(self, episode_index: int, video_key: str) -> Path:
                    """返回测试用video feature临时目录。"""
                    self.assert_inputs = (episode_index, video_key)
                    return video_directory

                def clear_episode_buffer(self, delete_images: bool = True) -> None:
                    """记录LeRobot缓冲区已被重置。"""
                    self.cleared = delete_images

            writer = LeRobotEpisodeWriter.__new__(LeRobotEpisodeWriter)
            writer.dataset = FakeDataset()
            writer.discard_episode()

            self.assertFalse(video_directory.exists())
            self.assertTrue(writer.dataset.cleared)
        finally:
            shutil.rmtree(temporary_directory, ignore_errors=True)

    def test_resume_rejects_mismatched_contract(self) -> None:
        """任一不可漂移字段变化都必须拒绝续采。"""
        actual = expected_contract()
        actual["fps"] = 60
        self.assertEqual(contract_mismatches(actual), ["fps"])


class DifferentialIKControllerTest(unittest.TestCase):
    """验证末端增量能转换为受限的绝对关节动作。"""

    def test_small_translation_produces_finite_absolute_action(self) -> None:
        """一毫米末端平移应收敛为有限七维绝对动作。"""
        with CleanTabletopEnv() as env:
            controller = DifferentialIKController(env)
            action, error = controller.command(
                TeleopDelta(
                    translation=np.array([0.001, 0.0, 0.0]),
                    rotation_rpy=np.zeros(3),
                    gripper=0.0,
                    meaningful=True,
                )
            )
            self.assertEqual(error, "")
            self.assertEqual(action.shape, (7,))
            self.assertTrue(np.isfinite(action).all())
            env.apply_joint_action(action, physics_steps=25)
            self.assertEqual(env.get_state().shape, (7,))


if __name__ == "__main__":
    unittest.main()
