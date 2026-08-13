#!/usr/bin/env bash
# 串联环境检查、单步训练与 checkpoint 完整性检查。

set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${SMOLVLA_PYTHON:-${PROJECT_ROOT}/.venv-cloud/bin/python}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "找不到云端 Python: ${PYTHON_BIN}；请先运行 bootstrap_cloud.sh。" >&2
  exit 1
fi

cd "${PROJECT_ROOT}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
exec "${PYTHON_BIN}" -m cloud.smoke_test "$@"
