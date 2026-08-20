<#
.SYNOPSIS
在 Windows 11 台式机（RTX 4060）上创建 smolvla-eval conda 环境并安装
SmolVLA 闭环评测依赖，pip 使用清华镜像以适配国内网络。

用法（在项目根目录 F:\桌面\smolvla 或 clone 目录）:
    powershell -ExecutionPolicy Bypass -File scripts\setup_eval_env.ps1
#>
$ErrorActionPreference = "Stop"

$envName = "smolvla-eval"
$pythonVersion = "3.10"
$pipMirror = "https://pypi.tuna.tsinghua.edu.cn/simple"
# 备用镜像（清华源大包同步慢时换用）:
# $pipMirror = "https://mirrors.aliyun.com/pypi/simple/"
# $pipMirror = "https://mirrors.cloud.tencent.com/pypi/simple/"

Write-Host "==> 1/4 创建 conda 环境: $envName (Python $pythonVersion)"
conda create -n $envName python=$pythonVersion pip -y
if ($LASTEXITCODE -ne 0) { throw "conda create 失败" }

Write-Host "==> 2/4 安装依赖（镜像: $pipMirror）"
conda run -n $envName python -m pip install -r requirements-eval.txt -i $pipMirror
if ($LASTEXITCODE -ne 0) { throw "pip install 失败" }

Write-Host "==> 3/4 验证关键包与 CUDA 可用性"
conda run -n $envName python -c 'import torch, mujoco, lerobot; print("torch", torch.__version__, "cuda_available", torch.cuda.is_available()); print("gpu", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO-GPU"); print("mujoco", mujoco.__version__); print("lerobot", lerobot.__version__)'
if ($LASTEXITCODE -ne 0) { throw "依赖验证失败" }

Write-Host "==> 4/4 完成。后续步骤:"
Write-Host "    conda activate $envName"
Write-Host "    python view_scene.py --headless --steps 10 --scene-seed 9   # MuJoCo 渲染冒烟"
Write-Host "    .\evaluate\run.ps1 --checkpoint <checkpoint目录> --config configs\eval_standard.yaml --output-dir outputs\eval\formal_020000 --resume"
