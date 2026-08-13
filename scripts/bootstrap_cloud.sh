#!/usr/bin/env bash
# 创建可复现的 Ubuntu SmolVLA 环境，并执行 GPU、模型和 EGL 预检。

set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${PROJECT_ROOT}/.venv-cloud"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://mirrors.nju.edu.cn/pytorch/whl/cu126}"
BOOTSTRAP_PYTHON="${SMOLVLA_BOOTSTRAP_PYTHON:-}"
INSTALL_SYSTEM_PACKAGES=0
SKIP_MODEL_DOWNLOAD=0

usage() {
  echo "用法: bash scripts/bootstrap_cloud.sh [--python PATH] [--venv PATH] [--torch-index-url URL] [--install-system-packages] [--skip-model-download]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv)
      VENV_PATH="$2"
      shift 2
      ;;
    --python)
      BOOTSTRAP_PYTHON="$2"
      shift 2
      ;;
    --torch-index-url)
      TORCH_INDEX_URL="$2"
      shift 2
      ;;
    --install-system-packages)
      INSTALL_SYSTEM_PACKAGES=1
      shift
      ;;
    --skip-model-download)
      SKIP_MODEL_DOWNLOAD=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "未知参数: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "${INSTALL_SYSTEM_PACKAGES}" == "1" ]]; then
  # 云容器经常直接以 root 运行且不提供 sudo；普通用户环境仍通过 sudo 提权。
  if [[ "$(id -u)" == "0" ]]; then
    APT_COMMAND=(apt-get)
  elif command -v sudo >/dev/null 2>&1; then
    APT_COMMAND=(sudo apt-get)
  else
    echo "安装系统依赖需要 root 或 sudo；请先安装 ffmpeg、libegl1、libgl1、libglvnd0 和 python3-venv。" >&2
    exit 1
  fi
  "${APT_COMMAND[@]}" update
  "${APT_COMMAND[@]}" install -y ffmpeg libegl1 libgl1 libglvnd0 python3-venv
fi

if [[ -z "${BOOTSTRAP_PYTHON}" ]] && command -v python3.11 >/dev/null 2>&1; then
  BOOTSTRAP_PYTHON="$(command -v python3.11)"
elif [[ -z "${BOOTSTRAP_PYTHON}" ]] && command -v python3.10 >/dev/null 2>&1; then
  BOOTSTRAP_PYTHON="$(command -v python3.10)"
elif [[ -z "${BOOTSTRAP_PYTHON}" ]] && command -v python3 >/dev/null 2>&1 && \
  python3 -c 'import sys; raise SystemExit(sys.version_info[:2] not in {(3, 10), (3, 11)})'; then
  BOOTSTRAP_PYTHON="$(command -v python3)"
elif [[ -z "${BOOTSTRAP_PYTHON}" ]] && command -v python >/dev/null 2>&1 && \
  python -c 'import sys; raise SystemExit(sys.version_info[:2] not in {(3, 10), (3, 11)})'; then
  BOOTSTRAP_PYTHON="$(command -v python)"
fi
if [[ -n "${BOOTSTRAP_PYTHON}" ]] && [[ ! -x "${BOOTSTRAP_PYTHON}" ]] && \
  command -v "${BOOTSTRAP_PYTHON}" >/dev/null 2>&1; then
  BOOTSTRAP_PYTHON="$(command -v "${BOOTSTRAP_PYTHON}")"
fi
if [[ -z "${BOOTSTRAP_PYTHON}" ]] || [[ ! -x "${BOOTSTRAP_PYTHON}" ]]; then
  echo "找不到 Python 3.10/3.11；请使用 --python 指向受支持的解释器。" >&2
  exit 1
fi
if ! "${BOOTSTRAP_PYTHON}" -c 'import sys; raise SystemExit(sys.version_info[:2] not in {(3, 10), (3, 11)})'; then
  echo "--python 必须指向 Python 3.10 或 3.11: ${BOOTSTRAP_PYTHON}" >&2
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "找不到 nvidia-smi；请先在云服务器安装兼容的 NVIDIA 驱动。" >&2
  exit 1
fi

"${BOOTSTRAP_PYTHON}" -m venv "${VENV_PATH}"
"${VENV_PATH}/bin/python" -m pip install --upgrade "pip==25.1.1" "setuptools==80.9.0" wheel
"${VENV_PATH}/bin/python" -m pip install \
  --index-url "${TORCH_INDEX_URL}" \
  "torch==2.7.0" "torchvision==0.22.0"
"${VENV_PATH}/bin/python" -m pip install \
  --constraint "${PROJECT_ROOT}/constraints.txt" \
  --requirement "${PROJECT_ROOT}/requirements-cloud.txt"

nvidia-smi
CHECK_ARGS=()
if [[ "${SKIP_MODEL_DOWNLOAD}" == "1" ]]; then
  CHECK_ARGS+=(--skip-model-download)
fi
cd "${PROJECT_ROOT}"
MUJOCO_GL=egl "${VENV_PATH}/bin/python" -m cloud.bootstrap_check "${CHECK_ARGS[@]}"

echo "云端环境准备完成。激活命令: source '${VENV_PATH}/bin/activate'"
