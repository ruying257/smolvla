<#
.SYNOPSIS
在本机 smolvla-eval Conda 环境中启动 SmolVLA 闭环评测。
#>

$ErrorActionPreference = "Stop"

if ($env:CONDA_DEFAULT_ENV -ne "smolvla-eval") {
    throw "当前 Conda 环境不是 smolvla-eval。请先执行: conda activate smolvla-eval"
}

$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $projectRoot
try {
    $env:HF_HUB_OFFLINE = "1"
    $env:TRANSFORMERS_OFFLINE = "1"
    & python -m evaluate @args
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}
finally {
    Pop-Location
}
