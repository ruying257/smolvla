# SmolVLA 本机模型效果评测

本文档定义 SmolVLA 在本笔记本 `smolvla-eval` Conda 环境中的 MuJoCo 闭环评测框架、命令、指标和验收口径。核心结论来自闭环任务成功率，不以训练 loss 或单步动作误差替代。

## 1. 评测框架

```text
完整 checkpoint
  -> 加载策略、预处理器和后处理器
  -> 按 scene_seed 重置 MuJoCo 场景
  -> 生成 canonical / unseen 任务指令
  -> 以 20 Hz 循环执行
       两路 256×256 RGB + 7 维状态
       -> SmolVLA CUDA 推理
       -> 7 维动作检查和限位
       -> MuJoCo 执行动作
       -> 严格成功/失败判定
  -> 输出逐条 CSV、汇总 JSON 和双相机 MP4
```

评测集合为 `scene_seeds × task_ids × prompt_types`：

- 四类任务：`red_on_blue`、`red_on_yellow`、`green_on_blue`、`green_on_yellow`；
- `canonical`：训练使用过的模板，如 `Put the red cube on the blue pad.`；
- `unseen`：训练未使用的同义模板，如 `Move the red cube to the blue pad.`；
- 同一 `scene_seed` 的积木初始位置严格复现。

本评测仅代表 MuJoCo 仿真闭环效果，不代表真实 UR10e 成功率。

## 2. 本机环境

已验证环境：Windows、Python 3.11、PyTorch 2.7.0+cu126、LeRobot 0.4.4、MuJoCo 3.6.0、NVIDIA GeForce GTX 1650。评测不使用云端 `.venv-cloud`，也不设置 EGL。

统一入口默认启用 Hugging Face 与 Transformers 离线模式，避免评测过程中访问网络。首次评测前，本机缓存必须包含 checkpoint 所引用的 `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` 配置、处理器和权重；缺失时入口会快速报错，需要先在可联网环境补齐缓存。

进入项目并激活环境：

```powershell
cd F:\桌面\smolvla
conda activate smolvla-eval
```

检查 Python、CUDA 和 GPU：

```powershell
python -c "import sys, torch; print(sys.version); print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

检查 MuJoCo 本机无窗口渲染：

```powershell
python view_scene.py --headless --steps 10 --scene-seed 10000
```

checkpoint 必须至少包含：

```text
pretrained_model/
├── config.json
├── model.safetensors
├── policy_preprocessor.json
└── policy_postprocessor.json
```

`--checkpoint` 可指向完整 `pretrained_model`、单个 checkpoint 或训练输出根目录，入口会自动寻找最新的完整模型。不要只复制 `model.safetensors`。

## 3. 执行命令

### 3.1 Windows 统一入口

PowerShell 入口会拒绝非 `smolvla-eval` 环境：

```powershell
conda activate smolvla-eval
.\evaluate\run.ps1 `
  --checkpoint outputs\train\smolvla_ur10e `
  --config configs\eval.yaml `
  --output-dir outputs\eval\basic
```

也可直接使用 Python 模块：

```powershell
python -m evaluate `
  --checkpoint outputs\train\smolvla_ur10e `
  --config configs\eval.yaml `
  --output-dir outputs\eval\basic
```

默认使用 `--device cuda`。仅在调试加载和控制链时使用 `--device cpu`。

本机 GTX 1650 的推理速度明显低于训练服务器。短链路通过只证明评测可运行，80条标准评测应预留较长时间，并以实际 `latency_mean_ms` 为准。

### 3.2 链路冒烟评测

只验证模型加载、CUDA 前向、渲染、动作执行及产物写出，不评价模型效果：

```powershell
.\evaluate\run.ps1 `
  --checkpoint outputs\train\smolvla_ur10e `
  --config configs\eval.yaml `
  --output-dir outputs\eval\smoke `
  --max-rollouts 1 `
  --max-steps 2
```

### 3.3 基础评测

`configs/eval.yaml` 包含 1 seed × 4 tasks × 1 canonical prompt，共 4 条 rollout：

```powershell
.\evaluate\run.ps1 `
  --checkpoint outputs\train\smolvla_ur10e `
  --config configs\eval.yaml `
  --output-dir outputs\eval\basic
```

基础评测用于快速回归，样本量不足以支持最终效果结论。

### 3.4 标准效果评测

`configs/eval_standard.yaml` 固定 10 seeds × 4 tasks × 2 prompts，共 80 条 rollout：

```powershell
.\evaluate\run.ps1 `
  --checkpoint outputs\train\smolvla_ur10e `
  --config configs\eval_standard.yaml `
  --output-dir outputs\eval\standard
```

正式运行前确认 `10000` 至 `10009` 未参与训练数据采集。若存在重合，应整体替换为一组固定的未见 seed，并对所有 checkpoint 使用同一集合。

### 3.5 调试覆盖参数

```powershell
python -m evaluate `
  --checkpoint outputs\train\smolvla_ur10e `
  --config configs\eval_standard.yaml `
  --output-dir outputs\eval\debug `
  --max-rollouts 2 `
  --max-steps 300
```

改变最大步数后的成功率不能与 600 步标准结果直接比较。

## 4. 指标口径

严格成功必须同时满足：指定积木完整进入目标区域内缩 5 mm 后的范围、连续稳定 0.5 秒、夹爪已经释放。

| 指标 | 定义 |
| --- | --- |
| 总体成功率 | `successes / rollouts`，第一主指标 |
| canonical / unseen 成功率 | 已见模板能力与语言措辞泛化能力 |
| 单任务成功率 | 分别检查四种颜色与目标组合 |
| 跨 seed 成功率 | 检查初始布局稳定性 |
| `steps` | 实际控制步数 |
| `latency_mean_ms` / `latency_p95_ms` | 单条轨迹的策略调用平均值与 P95 |
| `clipped_action_steps` | 至少一个动作维度被安全限位的步数 |

失败类型包括：

| `failure_mode` | 含义 |
| --- | --- |
| `success` | 满足严格成功条件 |
| `wrong_cube` | 操作错误积木 |
| `wrong_pad` | 放入错误区域 |
| `dropped_or_out_of_bounds` | 积木掉落或越界 |
| `timeout` | 最大步数内未完成 |
| `control_exception` | 状态、动作、推理、渲染或控制异常 |

当前汇总中的延迟 P95 是各 rollout P95 的再聚合，不是全部控制步合并后的全局 P95。不同 checkpoint 必须在相同代码、GPU、AMP、相机、FPS、步数和评测集合上比较。

## 5. 输出与检查

```text
outputs/eval/standard/
├── rollouts.csv
├── summary.json
└── videos/
    └── seed_<seed>_<task_id>_<prompt_type>.mp4
```

查看总体结果：

```powershell
Get-Content outputs\eval\standard\summary.json -Encoding UTF8
```

标准评测的 `rollouts.csv` 应有 80 条数据：

```powershell
$rows = Import-Csv outputs\eval\standard\rollouts.csv
$rows.Count
```

查看失败轨迹：

```powershell
$rows | Where-Object { $_.success -ne 'True' } | Format-Table scene_seed,task_id,prompt_type,failure_mode,video_path,error
```

按任务与措辞统计成功率：

```powershell
$rows | Group-Object task_id,prompt_type | ForEach-Object {
  $successes = @($_.Group | Where-Object { $_.success -eq 'True' }).Count
  [pscustomobject]@{ Group = $_.Name; Successes = $successes; Total = $_.Count; Rate = $successes / $_.Count }
} | Format-Table
```

检查动作裁剪或异常：

```powershell
$rows | Where-Object { [int]$_.clipped_action_steps -gt 0 -or $_.error } | Format-List
```

## 6. Checkpoint 公平对比

```powershell
python -m evaluate --checkpoint outputs\train\experiment_a --config configs\eval_standard.yaml --output-dir outputs\eval\experiment_a
python -m evaluate --checkpoint outputs\train\experiment_b --config configs\eval_standard.yaml --output-dir outputs\eval\experiment_b
```

先确认 rollout 数一致且 `error` 为空，再依次比较总体成功率、canonical/unseen 成功率、四类任务、各 seed、失败结构、成功轨迹步数、延迟和动作裁剪。不要依据单个视频、单个 seed 或训练 loss 宣布模型提升。

## 7. 验收清单

- [ ] 当前环境为 `smolvla-eval`，CUDA 可用；
- [ ] 使用完整 checkpoint 和训练时保存的处理器；
- [ ] 评测 seed 固定且不与训练采集 seed 重合；
- [ ] 正式评测覆盖四类任务和两种措辞；
- [ ] CSV 数量等于配置组合数，JSON 可解析，视频非空；
- [ ] `error` 为空，异常轨迹已修复后重新评测；
- [ ] checkpoint 对比使用完全相同的运行条件；
- [ ] 结论明确限定为 MuJoCo 仿真闭环结果。
