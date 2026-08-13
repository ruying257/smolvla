# SmolVLA 本机模型效果评测

本文档定义 SmolVLA 在本笔记本 `smolvla-eval` Conda环境中的MuJoCo闭环评测框架。正式结论来自固定实验矩阵上的重复闭环成功率，而不是训练loss、单个视频或单次随机rollout。

## 1. 为什么需要两种seed

评测包含两个相互独立的随机来源：

- `scene_seed`：控制两个积木的初始位置；同一seed复现相同场景。
- `policy_seed`：控制SmolVLA Flow Matching生成Action Chunk时的采样噪声。

只固定`scene_seed`不能复现完整策略轨迹。SmolVLA每次生成Action Chunk时会从随机噪声开始去噪；相同场景、任务和checkpoint可能因`policy_seed`不同而一次成功、一次失败。因此正式评测必须重复模型随机种子，并把两种seed都写入结果。

## 2. 正式实验矩阵

`configs/eval_standard.yaml`锁定：

```text
10个未见scene seed（10000-10009）
× 4类任务
× 3种措辞
× 2个policy seed（20260、20261）
= 240条rollout
```

三种措辞为：

| 类型 | 示例 | 训练状态 |
| --- | --- | --- |
| `canonical` | Put the red cube on the blue pad. | 已见 |
| `synonym` | Place the red cube onto the blue pad. | 已见 |
| `unseen` | Move the red cube to the blue pad. | 未见 |

训练数据使用的scene seed为`0-9`与`100-139`，正式测试使用`10000-10009`，不存在布局seed重合。专家示范最长351步，因此正式超时锁定为400步，即20 Hz下20秒，并提供约14%余量。

严格成功必须满足：指定积木完整进入目标区域内缩5 mm后的范围、连续稳定0.5秒且夹爪已经释放。

## 3. 环境与checkpoint

已验证环境为Windows、Python 3.11、PyTorch 2.7.0+cu126、LeRobot 0.4.4、MuJoCo 3.6.0和GTX 1650。评测不使用云端`.venv-cloud`或EGL。

```powershell
cd F:\桌面\smolvla
conda activate smolvla-eval
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
python view_scene.py --headless --steps 10 --scene-seed 10000
```

统一入口默认启用Hugging Face离线模式。本机缓存必须包含checkpoint引用的`HuggingFaceTB/SmolVLM2-500M-Video-Instruct`配置、处理器与权重。

checkpoint至少包含：

```text
pretrained_model/
├── config.json
├── model.safetensors
├── policy_preprocessor.json
└── policy_postprocessor.json
```

## 4. 分阶段执行

### 4.1 自动测试

```powershell
conda activate smolvla-eval
python -m unittest discover -s tests -v
```

### 4.2 两步链路冒烟

只验证checkpoint加载、CUDA前向、随机种子、视频、manifest和JSONL，不评价模型效果：

```powershell
.\evaluate\run.ps1 `
  --checkpoint outputs\train\smolvla_ur10e `
  --config configs\eval_standard.yaml `
  --output-dir outputs\eval\smoke_reproducible `
  --max-rollouts 4 `
  --max-steps 2 `
  --keep-all-videos
```

### 4.3 24条预实验

使用正式配置的第一个scene seed。由于矩阵顺序为`scene → task → prompt → policy`，前24条刚好覆盖：

```text
1 scene × 4 tasks × 3 prompts × 2 policy seeds
```

```powershell
.\evaluate\run.ps1 `
  --checkpoint outputs\train\smolvla_ur10e `
  --config configs\eval_standard.yaml `
  --output-dir outputs\eval\pilot_020000 `
  --max-rollouts 24 `
  --keep-all-videos
```

预实验用于检查显存、耗时、失败分类和恢复机制，不并入正式240条结果。

### 4.4 正式240条实验

正式运行开始后不得根据中间结果修改checkpoint、seed、措辞、超时或成功条件。

```powershell
.\evaluate\run.ps1 `
  --checkpoint outputs\train\smolvla_ur10e `
  --config configs\eval_standard.yaml `
  --output-dir outputs\eval\formal_020000
```

运行中断后使用完全相同的命令并增加`--resume`：

```powershell
.\evaluate\run.ps1 `
  --checkpoint outputs\train\smolvla_ur10e `
  --config configs\eval_standard.yaml `
  --output-dir outputs\eval\formal_020000 `
  --resume
```

默认在整批完成后保留全部失败视频，并为每个`task_id × prompt_type`保留按seed排序的第一条成功视频。若需要保留全部视频，首次启动和每次续跑都必须增加`--keep-all-videos`；视频策略属于manifest身份，不能在同一run中途改变。

## 5. 可恢复性与追溯

每条轨迹以以下字符串作为唯一实验键：

```text
scene=<scene_seed>|task=<task_id>|prompt=<prompt_type>|policy=<policy_seed>
```

每条完成后立即落盘到`rollouts.jsonl`并刷新磁盘。`--resume`会：

1. 校验checkpoint、权重SHA-256、评测源码SHA-256、环境、配置、步数、实验键和视频策略；
2. 拒绝损坏JSONL、重复实验键和缺失的保留视频；
3. 跳过合法完成项；
4. 移除并重新执行`control_exception`或带`error`的轨迹。

checkpoint、代码、环境或配置变化时必须创建新输出目录，不能向旧run混写。

## 6. 输出结构

```text
outputs/eval/formal_020000/
├── run_manifest.json       # checkpoint、代码、配置、环境和240个实验键
├── rollouts.jsonl          # 每条完成后即时追加的恢复日志
├── rollouts.csv            # 最终逐条结果
├── summary.json            # 机器可读统计
├── report.md               # 人工可读报告
├── video_retention.json    # 成功视频清理与保留清单
└── videos/
```

每条结果包含`scene_seed`、`policy_seed`、`rollout_key`、任务与措辞、成功和失败类型、步数、推理延迟、动作裁剪率、checkpoint SHA-256、视频状态、错误和完成时间。

## 7. 统计口径

第一主指标为总体严格成功率，同时报告：

- 按`scene_seed`整组重采样10000次得到的Bootstrap 95%置信区间；
- 四任务等权宏平均成功率；
- canonical、synonym、unseen成功率；
- `seen=(canonical+synonym)/2`与unseen的语言泛化差距；
- 分任务、分场景和分policy seed成功率；
- 固定场景、任务、措辞下的`2/2`稳定成功、`1/2`采样敏感、`0/2`稳定失败；
- `wrong_cube`、`wrong_pad`、`dropped_or_out_of_bounds`、`timeout`和`control_exception`分布；
- 成功轨迹步数中位数和P90；
- 发生裁剪的轨迹比例及总裁剪步数占比；
- 推理延迟中位数和P95。

`control_exception`属于评测有效性异常，入口返回非零状态；修复原因后通过`--resume`重跑，不能把它当作普通模型失败解释。

## 8. 结果检查

```powershell
$rows = Import-Csv outputs\eval\formal_020000\rollouts.csv
$rows.Count
($rows.rollout_key | Sort-Object -Unique).Count
Get-Content outputs\eval\formal_020000\summary.json -Encoding UTF8
Get-Content outputs\eval\formal_020000\report.md -Encoding UTF8
```

正式结果必须同时满足：240条结果、240个唯一实验键、10个scene seed、4类任务、3种措辞、2个policy seed、无`control_exception`。

本结果只代表当前checkpoint在本机MuJoCo仿真闭环中的能力，不代表真实UR10e成功率，也不能在缺少基座或其他checkpoint对照时宣称“训练带来提升”。
