"""与GUI无关的episode采集状态机。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CollectionPhase(str, Enum):
    """采集过程的三个互斥阶段。"""

    IDLE = "idle"
    RECORDING = "recording"
    PENDING_CONFIRMATION = "pending_confirmation"


@dataclass
class CollectionStateMachine:
    """管理自动开录、成功暂停以及人工保存或丢弃。"""

    phase: CollectionPhase = CollectionPhase.IDLE
    frame_count: int = 0

    def observe_action(self, meaningful: bool) -> bool:
        """处理一次控制动作并决定是否写入当前帧。

        Args:
            meaningful: 本次是否包含非零位姿增量或夹爪切换。

        Returns:
            当前帧是否应写入episode缓冲区。
        """
        if self.phase == CollectionPhase.IDLE and meaningful:
            self.phase = CollectionPhase.RECORDING
        if self.phase == CollectionPhase.RECORDING:
            self.frame_count += 1
            return True
        return False

    def observe_success(self, success: bool) -> bool:
        """在录制中检测到严格成功时进入人工确认阶段。

        Args:
            success: 环境严格成功信号。

        Returns:
            本次是否刚进入确认阶段。
        """
        if success and self.phase == CollectionPhase.RECORDING:
            self.phase = CollectionPhase.PENDING_CONFIRMATION
            return True
        return False

    def confirm(self) -> bool:
        """确认保存当前成功episode。

        Returns:
            处于确认阶段并接受确认时返回 ``True``。
        """
        if self.phase != CollectionPhase.PENDING_CONFIRMATION:
            return False
        self.reset()
        return True

    def discard(self) -> bool:
        """丢弃当前缓冲区并恢复空闲状态。

        Returns:
            丢弃前是否存在录制帧或待确认episode。
        """
        had_frames = self.phase != CollectionPhase.IDLE or self.frame_count > 0
        self.reset()
        return had_frames

    def reset(self) -> None:
        """恢复空闲且零帧的初始状态。"""
        self.phase = CollectionPhase.IDLE
        self.frame_count = 0

