"""杯子V3任务标识和唯一canonical训练指令。"""

from __future__ import annotations


TASK_PROMPTS = {
    "mug_on_blue": "Put the mug on the blue pad.",
    "mug_on_yellow": "Put the mug on the yellow pad.",
}
TASK_IDS = tuple(TASK_PROMPTS)
PROMPT_MODE = "canonical"


def task_prompt(task_id: str) -> str:
    """返回指定杯子任务唯一允许写入的数据集指令。

    Args:
        task_id: ``mug_on_blue``或``mug_on_yellow``。

    Returns:
        与任务标识严格对应的英文canonical指令。

    Raises:
        ValueError: 任务标识不属于V3锁定任务集合时抛出。
    """
    try:
        return TASK_PROMPTS[task_id]
    except KeyError as exc:
        raise ValueError(f"未知杯子任务: {task_id!r}，可选值={TASK_IDS}") from exc

