#!/usr/bin/env bash
# 汇总新云服务器的硬件、驱动和 SmolVLA 运行依赖，便于保存并回传诊断。

set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN=""
OUTPUT_PATH=""

usage() {
  echo "用法: bash scripts/check_server_environment.sh [--python PATH] [--output REPORT.txt]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --output)
      OUTPUT_PATH="$2"
      shift 2
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

resolve_python() {
  # 按“显式参数、项目虚拟环境、当前镜像解释器”的顺序选择 Python。
  if [[ -n "${PYTHON_BIN}" ]]; then
    if [[ -x "${PYTHON_BIN}" ]]; then
      return
    fi
    if command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
      PYTHON_BIN="$(command -v "${PYTHON_BIN}")"
      return
    fi
    PYTHON_BIN=""
    return
  fi

  if [[ -x "${PROJECT_ROOT}/.venv-cloud/bin/python" ]]; then
    PYTHON_BIN="${PROJECT_ROOT}/.venv-cloud/bin/python"
    return
  fi
  local candidate
  for candidate in python python3.11 python3.10 python3; do
    if command -v "${candidate}" >/dev/null 2>&1; then
      PYTHON_BIN="$(command -v "${candidate}")"
      return
    fi
  done
}

section() {
  # 使用稳定标题分隔报告，方便从长日志中定位异常部分。
  printf '\n===== %s =====\n' "$1"
}

command_or_missing() {
  # 执行可选命令；缺失时继续报告其他环境信息。
  local command_name="$1"
  shift
  if command -v "${command_name}" >/dev/null 2>&1; then
    "${command_name}" "$@" 2>&1 || true
  else
    echo "MISSING: ${command_name}"
  fi
}

quick_check() {
  # 输出一页式关键门禁，不替代 bootstrap_cloud.sh 的真实模型与 EGL 检查。
  local label="$1"
  local status="$2"
  printf '%-28s %s\n' "${label}" "${status}"
}

generate_report() {
  section "基本信息"
  echo "report_time: $(date --iso-8601=seconds 2>/dev/null || date)"
  echo "hostname: $(hostname 2>/dev/null || echo UNKNOWN)"
  echo "user: $(id -un 2>/dev/null || echo UNKNOWN)"
  echo "working_directory: $(pwd)"
  echo "project_root: ${PROJECT_ROOT}"
  echo "shell: ${SHELL:-UNKNOWN}"
  command_or_missing uname -a

  section "操作系统"
  if [[ -r /etc/os-release ]]; then
    grep -E '^(PRETTY_NAME|NAME|VERSION|VERSION_ID|ID)=' /etc/os-release || true
  else
    echo "MISSING: /etc/os-release"
  fi

  section "CPU"
  if command -v lscpu >/dev/null 2>&1; then
    lscpu | grep -E '^(Architecture|CPU\(s\)|On-line CPU|Model name|Thread|Core|Socket|NUMA)' || true
  else
    echo "MISSING: lscpu"
  fi

  section "内存"
  command_or_missing free -h

  section "磁盘"
  command_or_missing df -h "${PROJECT_ROOT}"
  if command -v du >/dev/null 2>&1; then
    du -sh "${PROJECT_ROOT}" 2>/dev/null || true
    if [[ -d "${PROJECT_ROOT}/smolvla-data" ]]; then
      du -sh "${PROJECT_ROOT}/smolvla-data" 2>/dev/null || true
    fi
  fi

  section "NVIDIA GPU 与驱动"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi 2>&1 || true
    echo
    nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.free,temperature.gpu,pstate \
      --format=csv,noheader 2>&1 || true
  else
    echo "MISSING: nvidia-smi"
  fi
  ls -l /dev/nvidia* 2>/dev/null || echo "MISSING: /dev/nvidia*"

  section "CUDA 工具链"
  command_or_missing nvcc --version
  echo "CUDA_HOME: ${CUDA_HOME:-UNSET}"
  echo "PATH_has_cuda: $(printf '%s' "${PATH:-}" | grep -o '/[^:]*cuda[^:]*' | paste -sd ',' - || true)"

  section "Python 与机器学习依赖"
  if [[ -n "${PYTHON_BIN}" ]] && [[ -x "${PYTHON_BIN}" ]]; then
    echo "python_executable: ${PYTHON_BIN}"
    "${PYTHON_BIN}" - <<'PY'
import importlib.metadata
import platform
import sys

print("python_version:", platform.python_version())
print("python_full:", sys.version.replace("\n", " "))
packages = (
    "torch",
    "torchvision",
    "torchcodec",
    "lerobot",
    "mujoco",
    "transformers",
    "datasets",
    "accelerate",
    "av",
    "imageio-ffmpeg",
)
for package in packages:
    try:
        version = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        version = "MISSING"
    print(f"package_{package}: {version}")

try:
    import torch

    print("torch_version:", torch.__version__)
    print("torch_cuda_runtime:", torch.version.cuda)
    print("torch_cuda_available:", torch.cuda.is_available())
    print("cudnn_version:", torch.backends.cudnn.version())
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            print(f"gpu_{index}_name: {properties.name}")
            print(f"gpu_{index}_vram_gib: {properties.total_memory / 1024**3:.2f}")
            print(f"gpu_{index}_compute_capability: {properties.major}.{properties.minor}")
except Exception as exc:
    print(f"torch_probe_error: {type(exc).__name__}: {exc}")
PY
  else
    echo "MISSING: 可用 Python 解释器"
  fi

  section "视频与图形运行库"
  command_or_missing ffmpeg -version
  echo "MUJOCO_GL: ${MUJOCO_GL:-UNSET}"
  if command -v ldconfig >/dev/null 2>&1; then
    ldconfig -p 2>/dev/null | grep -E 'lib(EGL|GLX|GL|cuda)\.so' | head -n 40 || true
  else
    echo "MISSING: ldconfig"
  fi

  section "项目数据与关键文件"
  for relative_path in \
    assets/mujoco/scene.xml \
    requirements-cloud.txt \
    constraints.txt \
    configs/cloud_train.yaml \
    configs/cloud_eval.yaml \
    smolvla-data/smolvla_ur10e/meta \
    smolvla-data/smolvla_ur10e/data \
    smolvla-data/smolvla_ur10e/videos; do
    if [[ -e "${PROJECT_ROOT}/${relative_path}" ]]; then
      echo "OK: ${relative_path}"
    else
      echo "MISSING: ${relative_path}"
    fi
  done
  if command -v git >/dev/null 2>&1 && git -C "${PROJECT_ROOT}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "git_commit: $(git -C "${PROJECT_ROOT}" rev-parse HEAD 2>/dev/null || echo UNKNOWN)"
    echo "git_branch: $(git -C "${PROJECT_ROOT}" branch --show-current 2>/dev/null || echo UNKNOWN)"
  fi

  section "快速结论"
  if command -v nvidia-smi >/dev/null 2>&1; then
    quick_check "NVIDIA driver" "OK"
  else
    quick_check "NVIDIA driver" "MISSING"
  fi
  if command -v ffmpeg >/dev/null 2>&1; then
    quick_check "FFmpeg" "OK"
  else
    quick_check "FFmpeg" "MISSING"
  fi
  if [[ -n "${PYTHON_BIN}" ]] && [[ -x "${PYTHON_BIN}" ]]; then
    quick_check "Python" "OK (${PYTHON_BIN})"
    if "${PYTHON_BIN}" -c 'import torch; raise SystemExit(not torch.cuda.is_available())' >/dev/null 2>&1; then
      quick_check "PyTorch CUDA" "OK"
    else
      quick_check "PyTorch CUDA" "MISSING/FAILED"
    fi
  else
    quick_check "Python" "MISSING"
    quick_check "PyTorch CUDA" "NOT CHECKED"
  fi
  if [[ -d "${PROJECT_ROOT}/smolvla-data/smolvla_ur10e" ]]; then
    quick_check "SmolVLA dataset root" "OK"
  else
    quick_check "SmolVLA dataset root" "MISSING"
  fi
  echo
  echo "说明：本报告只查询环境，不安装依赖、不下载模型、不修改系统。"
}

resolve_python
if [[ -n "${OUTPUT_PATH}" ]]; then
  if [[ "${OUTPUT_PATH}" != /* ]]; then
    OUTPUT_PATH="${PROJECT_ROOT}/${OUTPUT_PATH}"
  fi
  mkdir -p "$(dirname "${OUTPUT_PATH}")"
  generate_report | tee "${OUTPUT_PATH}"
  echo "环境报告已保存: ${OUTPUT_PATH}"
else
  generate_report
fi
