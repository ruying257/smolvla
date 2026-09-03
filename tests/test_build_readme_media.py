"""验证 README 可视化素材生成工具。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import av
import numpy as np
from PIL import Image

from scripts.build_readme_media import (
    DomainEpisode,
    TRAJECTORY_CASES,
    _agent_view,
    _compose_grid,
    decode_video_segment,
    resolve_chinese_font,
    save_gif_with_budget,
    select_domain_pairs,
)


class ReadmeMediaTest(unittest.TestCase):
    """覆盖素材布局、同步和输出约束。"""

    def test_compose_grid_uses_requested_columns(self) -> None:
        """四张等尺寸图应组成正确的 2×2 宫格。"""

        images = [Image.new("RGB", (40, 30), (index * 20, 0, 0)) for index in range(4)]
        result = _compose_grid(images, columns=2)
        self.assertEqual(result.size, (80, 60))

    def test_trajectory_cases_cover_four_paired_rollouts(self) -> None:
        """轨迹对照应固定为四组且覆盖蓝黄两类任务。"""

        self.assertEqual(len(TRAJECTORY_CASES), 4)
        self.assertEqual(len({case.video_stem for case in TRAJECTORY_CASES}), 4)
        self.assertEqual(
            {case.task_id for case in TRAJECTORY_CASES},
            {"mug_on_blue", "mug_on_yellow"},
        )

    def test_agent_view_crops_left_camera_from_evaluation_video(self) -> None:
        """双视角评测帧应只保留左侧第三视角。"""

        frame = np.zeros((32, 64, 3), dtype=np.uint8)
        frame[:, 32:] = 255
        result = _agent_view(frame)
        self.assertEqual(result.shape, (32, 32, 3))
        self.assertEqual(int(result.max()), 0)

    def test_select_domain_pairs_keeps_exact_source_match(self) -> None:
        """原始域与随机化域必须按同一源 episode 严格配对。"""

        episodes = [
            self._domain_episode(0, 3, "original", "original", "default"),
            self._domain_episode(1, 3, "dr0", "changed", "alt"),
            self._domain_episode(2, 7, "original", "original", "default"),
            self._domain_episode(3, 7, "dr0", "green_white", "default"),
        ]
        pairs = select_domain_pairs(episodes, [7, 3])
        self.assertEqual([(left.source_episode, right.source_episode) for left, right in pairs], [(7, 7), (3, 3)])
        self.assertTrue(all(left.frame_count == right.frame_count for left, right in pairs))

    def test_select_domain_pairs_rejects_frame_mismatch(self) -> None:
        """物理一致配对的帧数不一致时应拒绝生成。"""

        original = self._domain_episode(0, 3, "original", "original", "default")
        randomized = DomainEpisode(
            episode_index=1,
            source_episode=3,
            variant="dr0",
            texture="changed",
            lighting="alt",
            scene_seed=123,
            task_id="mug_on_blue",
            frame_count=11,
            max_state_dev=0.001,
        )
        with self.assertRaisesRegex(ValueError, "帧数不一致"):
            select_domain_pairs([original, randomized], [3])

    def test_decode_video_segment_respects_time_bounds(self) -> None:
        """合并视频解码应只返回指定时间范围内的帧。"""

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.mp4"
            self._write_test_video(path, frame_count=10, fps=10)
            frames = decode_video_segment(path, start_seconds=0.2, end_seconds=0.5)
        self.assertEqual(len(frames), 3)
        self.assertLess(int(frames[0][0, 0, 0]), int(frames[-1][0, 0, 0]))

    def test_save_gif_with_budget_creates_looping_animation(self) -> None:
        """GIF 应循环播放并满足给定文件大小上限。"""

        frames = [Image.new("RGB", (128, 64), (index * 30, 50, 100)) for index in range(6)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.gif"
            save_gif_with_budget(frames, path, fps=8, max_megabytes=0.2, minimum_width=32)
            with Image.open(path) as image:
                self.assertEqual(image.n_frames, 6)
                self.assertEqual(image.info.get("loop"), 0)
            self.assertLessEqual(path.stat().st_size, int(0.2 * 1024 * 1024))

    def test_explicit_missing_font_does_not_silently_fallback(self) -> None:
        """用户指定的字体无效时应给出明确错误。"""

        with self.assertRaisesRegex(FileNotFoundError, "指定的中文字体不存在"):
            resolve_chinese_font(Path("definitely_missing_chinese_font.ttf"))

    @staticmethod
    def _domain_episode(
        episode_index: int,
        source_episode: int,
        variant: str,
        texture: str,
        lighting: str,
    ) -> DomainEpisode:
        """构造测试使用的域随机化 episode。"""

        return DomainEpisode(
            episode_index=episode_index,
            source_episode=source_episode,
            variant=variant,
            texture=texture,
            lighting=lighting,
            scene_seed=123,
            task_id="mug_on_blue",
            frame_count=10,
            max_state_dev=0.001,
        )

    @staticmethod
    def _write_test_video(path: Path, frame_count: int, fps: int) -> None:
        """写入具有可区分灰度值的短测试视频。"""

        with av.open(str(path), "w") as container:
            stream = container.add_stream("mpeg4", rate=fps)
            stream.width = 64
            stream.height = 64
            stream.pix_fmt = "yuv420p"
            for index in range(frame_count):
                image = np.full((64, 64, 3), index * 20, dtype=np.uint8)
                frame = av.VideoFrame.from_ndarray(image, format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)


if __name__ == "__main__":
    unittest.main()
