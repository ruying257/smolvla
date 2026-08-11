#!/usr/bin/env bash
# 从项目根目录调用配置驱动的 SmolVLA 训练入口。

set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${SMOLVLA_PYTHON:-${PROJECT_ROOT}/.venv-cloud/bin/python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "找不到云端 Python: ${PYTHON_BIN}；请先运行 bootstrap_cloud.sh。" >&2
  exit 1
fi

cd "${PROJECT_ROOT}"
exec "${PYTHON_BIN}" -m cloud.train "$@"
