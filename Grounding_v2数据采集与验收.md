# SmolVLA 颜色 Grounding v2 数据采集与验收

本流程只采集、复核和验收数据，不启动训练。最终目录固定为：

```text
smolvla-data/smolvla_ur10e_grounding_v2
```

采集期间，每个队列键单独保存在隐藏工作目录
`smolvla-data/.smolvla_ur10e_grounding_v2_collection`。只有80条全部完成、
20个蒙太奇全部人工标记为 `pass` 且自动验收通过后，工具才会物化最终数据集。
被重采的旧分片会归档，不会进入最终80条。

## 0. 环境

本机已验证可用的环境是 `smolvla-collector-clean`：

```powershell
conda activate smolvla-collector-clean
python -m pip install -r requirements-collector.txt
```

## 1. 采集8条pilot

首次启动：

```powershell
python -m collector.collect_matrix `
  --config configs\collect_grounding_v2.yaml `
  --pilot
```

若中途关闭窗口，未按 Enter 的episode不会保存。继续pilot时必须显式恢复：

```powershell
python -m collector.collect_matrix `
  --config configs\collect_grounding_v2.yaml `
  --pilot `
  --resume
```

同一scene的四个任务会复用 `initial_references/scene_<seed>.npz` 中的无损
动作前首帧。该设计避免OpenGL重复渲染的细微像素波动破坏原始图像哈希，
并保证中断恢复后初始机器人状态、积木位姿和两路图像仍完全一致。

如果工作区由旧版采集器创建，已经保存了分片但还没有无损基准，工具会拒绝
伪造哈希。先归档并重采提示中的已有键，例如：

```powershell
python -m collector.collect_matrix `
  --config configs\collect_grounding_v2.yaml `
  --resume `
  --redo-key "scene=210|task=red_on_blue|prompt=canonical"
```

然后继续pilot：

```powershell
python -m collector.collect_matrix `
  --config configs\collect_grounding_v2.yaml `
  --pilot `
  --resume
```

操作必须遵守：接近目标积木、对准、闭合、抬升、移动、下降、释放、稳定确认。
不要长时间停顿、绕行、碰错误积木、反复抓取或接受明显IK抖动。达到400帧会自动
丢弃，并停留在当前队列键重试。

## 2. 生成并复核pilot蒙太奇

```powershell
python -m collector.build_review_montages `
  --config configs\collect_grounding_v2.yaml `
  --pilot
```

观看下面两个视频：

```text
smolvla-data/.smolvla_ur10e_grounding_v2_collection/review_montages/scene_210.mp4
smolvla-data/.smolvla_ur10e_grounding_v2_collection/review_montages/scene_212.mp4
```

在 `review_status.csv` 中把scene 210和212改为 `pass`。若某条有问题，先执行本页
“局部重采”命令，不得直接标记为pass。

完成复核后运行pilot验收：

```powershell
python -m collector.validate_grounding_dataset `
  --config configs\collect_grounding_v2.yaml `
  --pilot
```

只有退出码为0且 `pilot_validation.json.status` 为 `pass`，全量续采才会开放。

## 3. 继续采集剩余72条

```powershell
python -m collector.collect_matrix `
  --config configs\collect_grounding_v2.yaml `
  --resume
```

任何中断都重复同一条命令。工具会逐条解码检查已完成分片，只跳过契约、Parquet、
双路视频和帧数均完整的键；当前键失败不会跳到下一个任务。

## 4. 生成并复核全部20个蒙太奇

默认2倍速：

```powershell
python -m collector.build_review_montages `
  --config configs\collect_grounding_v2.yaml
```

也可指定2至4倍速：

```powershell
python -m collector.build_review_montages `
  --config configs\collect_grounding_v2.yaml `
  --speed 4
```

逐一观看20个视频，并把 `review_status.csv` 中对应scene改为 `pass`。每个画面固定为：

```text
red_on_blue    | green_on_blue
red_on_yellow  | green_on_yellow
```

## 5. 局部重采

例如只重采scene 210的红积木到蓝板：

```powershell
python -m collector.collect_matrix `
  --config configs\collect_grounding_v2.yaml `
  --resume `
  --redo-key "scene=210|task=red_on_blue|prompt=canonical"
```

重采后重新生成蒙太奇、重新观看该scene，并再次把它标记为 `pass`。旧分片只进入
`archived_shards`，不会进入最终数据集。

## 6. 全量验收并生成最终数据集

```powershell
python -m collector.validate_grounding_dataset `
  --config configs\collect_grounding_v2.yaml `
  --finalize
```

该命令会先验收80个分片，再按固定队列顺序重建最终LeRobot数据集，最后重新读取
全部Parquet和两路视频。成功时必须同时存在：

```text
smolvla-data/smolvla_ur10e_grounding_v2/collection_manifest.json
smolvla-data/smolvla_ur10e_grounding_v2/dataset_validation.json
smolvla-data/smolvla_ur10e_grounding_v2/grounding_v2_eda_report.md
smolvla-data/smolvla_ur10e_grounding_v2/dataset_sha256.txt
```

查看最终结论：

```powershell
Get-Content -Raw -Encoding UTF8 `
  smolvla-data\smolvla_ur10e_grounding_v2\dataset_validation.json
```

只有其中 `status` 等于 `pass` 时，才能把该目录交给训练任务。
