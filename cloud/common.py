"""云端训练入口共享的配置、路径与日志工具。"""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path
from typing import Any, Sequence

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_yaml_config(path: Path) -> dict[str, Any]:
    """读取 YAML 配置并要求根节点为映射。

    Args:
        path: YAML 配置文件路径。

    Returns:
        配置字典。
    """
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"配置文件不存在: {resolved}")
    content = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(content, dict):
        raise ValueError(f"配置根节点必须是映射: {resolved}")
    return content


def resolve_path(value: str | Path, base: Path = PROJECT_ROOT) -> Path:
    """把相对路径解析为相对于项目根目录的绝对路径。

    Args:
        value: 绝对路径、相对路径或包含 ``~`` 的路径。
        base: 相对路径的解析基准。

    Returns:
        解析后的绝对路径。
    """
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def format_command(command: Sequence[str]) -> str:
    """把参数列表格式化为可复制的 POSIX 命令。"""
    return shlex.join(str(item) for item in command)


def run_logged(command: Sequence[str], log_path: Path, cwd: Path = PROJECT_ROOT) -> int:
    """执行命令，并把合并后的输出同时写入终端和日志。

    Args:
        command: 子进程参数列表。
        log_path: UTF-8 日志文件路径。
        cwd: 子进程工作目录。

    Returns:
        子进程退出码。
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", newline="\n") as log_file:
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()
        return process.wait()
