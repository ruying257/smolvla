"""V1/V2 积木任务定义的兼容入口。

代码清理把积木任务定义移入 ``collector/common/legacy/task_spec.py``，
但 ``evaluate/rollout`` 及其评测链路仍通过 ``collector.common.task_spec``
引用 ``TASKS``/``CollectionTask``。本模块保持该公共路径可用，避免改动
评测模块的导入与源码身份校验逻辑。
"""

from __future__ import annotations

from collector.common.legacy.task_spec import (  # noqa: F401
    TASKS,
    CollectionTask,
    choose_balanced_template,
)
