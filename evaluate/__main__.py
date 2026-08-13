"""支持通过 ``python -m evaluate`` 启动本机离线模型评测。"""

import os


os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from evaluate.rollout import main


if __name__ == "__main__":
    raise SystemExit(main())
