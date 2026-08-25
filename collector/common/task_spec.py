"""V1/V2 积木任务定义（供 ``evaluate/rollout`` 评测链路继续引用）。

定义曾位于 ``collector/common/legacy/task_spec.py``，但 ``collector/common/legacy/``
被 ``.gitignore`` 的 ``*/legacy/`` 规则忽略，git 同步的机器上没有该目录。为避免
评测在 git 同步环境（如实验室主机）导入失败，本模块改为**自包含**内嵌这些定义，
不再依赖被忽略的 legacy 路径。语义与既有 ``evaluate/rollout`` 的引用完全一致：
``TASKS``（四类积木任务）、``CollectionTask``、``choose_balanced_template``。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CollectionTask:
    """描述一个固定的积木到放置区任务。

    Attributes:
        task_id: 稳定的内部任务标识。
        cube_color: 英文积木颜色。
        pad_color: 英文放置区颜色。
    """

    task_id: str
    cube_color: str
    pad_color: str

    def prompt(self, template_id: str) -> str:
        """按模板生成英文任务文本。

        Args:
            template_id: ``canonical`` 或 ``synonym``。

        Returns:
            可直接写入LeRobot数据集的英文指令。

        Raises:
            ValueError: 模板标识未知时抛出。
        """
        if template_id == "canonical":
            return f"Put the {self.cube_color} cube on the {self.pad_color} pad."
        if template_id == "synonym":
            return f"Place the {self.cube_color} cube onto the {self.pad_color} pad."
        raise ValueError(f"未知 template_id={template_id!r}")


TASKS = {
    "red_on_blue": CollectionTask("red_on_blue", "red", "blue"),
    "red_on_yellow": CollectionTask("red_on_yellow", "red", "yellow"),
    "green_on_blue": CollectionTask("green_on_blue", "green", "blue"),
    "green_on_yellow": CollectionTask("green_on_yellow", "green", "yellow"),
}


def choose_balanced_template(task_id: str, counts: dict[str, int]) -> str:
    """为指定任务选择当前数量较少的训练措辞。

    Args:
        task_id: 四类任务之一。
        counts: 键格式为 ``task_id/template_id`` 的已保存episode计数。

    Returns:
        ``canonical`` 或 ``synonym``；数量相同时优先canonical。
    """
    if task_id not in TASKS:
        raise ValueError(f"未知 task_id={task_id!r}")
    canonical_count = counts.get(f"{task_id}/canonical", 0)
    synonym_count = counts.get(f"{task_id}/synonym", 0)
    return "canonical" if canonical_count <= synonym_count else "synonym"
