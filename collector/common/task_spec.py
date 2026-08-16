"""四类语言条件任务及训练措辞定义。"""

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

