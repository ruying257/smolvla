"""验证 ChunkBlendPolicy 的 chunk 衔接平滑（角度回卷 + 夹爪保护）。"""

from __future__ import annotations

import unittest

import numpy as np

from evaluate.rollout import ChunkBlendPolicy, _tensor_to_numpy_action_chunk, _wrap_angle


class _FakePolicy:
    """最小桩策略：predict_action_chunk 顺序返回预设的整段 chunk。"""

    def __init__(self, chunks: list[np.ndarray], n_action_steps: int = 5) -> None:
        self.chunks = [np.asarray(c, dtype=np.float64) for c in chunks]
        self.n_action_steps = n_action_steps
        self.reset_calls = 0

    @property
    def config(self):
        return SimpleConfig(self.n_action_steps)

    def reset(self) -> None:
        self.reset_calls += 1

    def predict_action_chunk(self, batch):
        if self.chunks:
            return self.chunks.pop(0)
        return np.zeros((self.n_action_steps, 7))


class SimpleConfig:
    def __init__(self, n_action_steps: int) -> None:
        self.n_action_steps = n_action_steps


def _uniform(value: float, steps: int = 5) -> np.ndarray:
    """构造每一步前 6 维为 value、夹爪为 0 的 chunk。"""
    return np.full((steps, 7), value) * np.array([1.0] * 6 + [0.0])


class ChunkBlendPolicyTest(unittest.TestCase):
    def _new(self, chunks, n_action_steps=5, k=3):
        return ChunkBlendPolicy(_FakePolicy(chunks, n_action_steps), k)

    def test_wrap_angle(self):
        """角度差回卷到 [-pi, pi)，跨 π 走最短弧。"""
        # 3.0 与 -3.0 的差为 -6.0，最短弧为 +0.283（3.0 沿正方向到 -3.0）。
        self.assertAlmostEqual(float(_wrap_angle(np.array([-6.0]))[0]), 0.283185307179586, places=5)
        # 已在 [-pi, pi) 内的差保持不变。
        self.assertAlmostEqual(float(_wrap_angle(np.array([0.5]))[0]), 0.5)
        # 邻近 π 的差值仍收敛到短弧一侧。
        self.assertAlmostEqual(float(_wrap_angle(np.array([3.2]))[0]), 3.2 - 2 * np.pi, places=5)

    def test_blend_reduces_first_frame_jump(self):
        """边界首帧被拉向旧 chunk 尾帧锚点，削减输出跳变。"""
        old_chunk = _uniform(0.0)      # 旧 chunk 前 6 维全 0
        new_chunk = _uniform(1.0)      # 新 chunk 前 6 维全 1（跳变 1 rad）
        policy = self._new([old_chunk, new_chunk], k=3)
        policy.reset()
        for _ in range(5):            # 出队旧 chunk
            policy.select_action({})
        blended_first = policy.select_action({})   # 触发重预测 + blend
        # K=3 首帧权重 1/3：anchor=0 + (1/3)*1
        self.assertAlmostEqual(float(blended_first[0]), 1.0 / 3.0, places=5)
        # 跳变应从 1.0 削减到 0.333
        self.assertLess(float(blended_first[0]), 1.0)

    def test_angle_wrap_prevents_long_way(self):
        """跨 π 时 blend 沿短弧走，不绕一整圈。"""
        old_chunk = _uniform(0.0)
        # 新 chunk 关节角从接近 +π 跳到接近 -π（差 ≈ -6.28）。
        new_chunk = _uniform(0.0)
        new_chunk[:, 0] = -3.0
        new_chunk[:, 0] += 3.14159  # 旧尾 0，新首 ≈ +0.14（几乎未跳）
        # 让锚点接近 +3.0：构造一个旧尾为 +3.0 的旧 chunk 再人为替换。
        old_chunk[-1, 0] = 3.0
        new_chunk[:, 0] = -3.0      # 新 chunk 首帧在 -π 另一侧
        policy = self._new([old_chunk, new_chunk], k=3)
        policy.reset()
        for _ in range(5):
            policy.select_action({})
        blended_first = policy.select_action({})[0]
        # 短弧：从 3.0 到 -3.0 只应移动 ≈0.28 rad，回卷后首帧仍在 π 附近。
        self.assertGreater(float(blended_first), 2.5)

    def test_gripper_dimension_not_blended(self):
        """夹爪维度（第7维）应透传新 chunk 值，不做插值。"""
        old_chunk = np.zeros((5, 7))
        old_chunk[:, 6] = 1.0       # 旧 chunk 夹爪闭合
        new_chunk = np.zeros((5, 7))
        new_chunk[:, 6] = 0.0       # 新 chunk 夹爪张开
        policy = self._new([old_chunk, new_chunk], k=3)
        policy.reset()
        for _ in range(5):
            policy.select_action({})
        first = policy.select_action({})
        self.assertEqual(float(first[6]), 0.0)

    def test_blend_only_affects_leading_frames(self):
        """chunk 前 K 帧之后的帧保持模型原始输出。"""
        old_chunk = _uniform(0.0)
        new_chunk = np.arange(5 * 7, dtype=np.float64).reshape(5, 7)
        policy = self._new([old_chunk, new_chunk.copy()], n_action_steps=5, k=2)
        policy.reset()
        for _ in range(5):
            policy.select_action({})
        # 触发重预测 + blend 前 2 帧
        policy.select_action({})
        policy.select_action({})
        for i in range(2, 5):       # 第 3~5 帧应为模型原始值
            frame = policy.select_action({})
            np.testing.assert_allclose(frame, new_chunk[i])

    def test_k_zero_is_pass_through(self):
        """blend_frames=0 时应逐帧透传，不改变输出。"""
        chunk = np.arange(5 * 7, dtype=np.float64).reshape(5, 7)
        policy = self._new([chunk.copy()], n_action_steps=5, k=0)
        policy.reset()
        for i in range(5):
            np.testing.assert_allclose(policy.select_action({}), chunk[i])

    def test_horizon_truncation(self):
        """预测出的 chunk 应按 n_action_steps 截断。"""
        big_chunk = np.arange(8 * 7, dtype=np.float64).reshape(8, 7)
        fake = _FakePolicy([big_chunk], n_action_steps=3)
        policy = ChunkBlendPolicy(fake, blend_frames=2)
        policy.reset()
        frames = [policy.select_action({}) for _ in range(3)]
        self.assertEqual(len(frames), 3)
        np.testing.assert_allclose(frames[2], big_chunk[2])

    def test_reset_clears_state(self):
        """reset 应清空队列与衔接状态，避免跨 rollout 污染。"""
        chunk = _uniform(0.0)
        fake = _FakePolicy([chunk.copy(), chunk.copy()], n_action_steps=5)
        policy = ChunkBlendPolicy(fake, blend_frames=2)
        policy.reset()
        for _ in range(5):
            policy.select_action({})
        policy.reset()
        self.assertEqual(policy._queue, [])
        self.assertIsNone(policy._prev_chunk)
        self.assertGreaterEqual(fake.reset_calls, 2)

    def test_tensor_to_numpy_action_chunk(self):
        """应把 (batch, steps, dim) 收缩为 (steps, dim)。"""
        arr = _tensor_to_numpy_action_chunk(np.zeros((1, 5, 7)))
        self.assertEqual(arr.shape, (5, 7))


if __name__ == "__main__":
    unittest.main()
