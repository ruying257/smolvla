# SmolVLA 杯子 V3 数据采集与验收

本流程只负责杯子双目标区域的人工专家数据采集、回放和验收，不启动训练，
也不会自动操纵机器人。全部命令固定使用已有的 `smolvla-collector-clean` 环境。

最终数据集目录：

```text
smolvla-data/smolvla_ur10e_mug_v3
```

采集中的单episode原子分片位于：

```text
smolvla-data/.smolvla_ur10e_mug_v3_collection
```

## 1. 复核固定seed筛选结果

该命令实际扫描 `0..9999`，用 `MugTabletopEnv.reset(seed)` 的稳定杯子位置
重新生成4×5覆盖报告，并验证结果与锁定配置逐项一致。运行约需数分钟。

```powershell
conda activate smolvla-collector-clean
python -m collector.v3.select_seeds `
  --verify-config configs\collect_mug_v3.yaml
```

固定结果保存在：

```text
configs/mug_v3_seed_selection.json
configs/mug_v3_seed_selection.csv
```

## 2. 人工采集4条pilot

第一次启动：

```powershell
conda activate smolvla-collector-clean
python -m collector.v3.collect_matrix `
  --config configs\collect_mug_v3.yaml `
  --pilot
```

中断后继续：

```powershell
python -m collector.v3.collect_matrix `
  --config configs\collect_mug_v3.yaml `
  --pilot `
  --resume
```

pilot固定为seed `4164`、`2337`的蓝黄配对，共4条。操作协议为：接近杯子、
对准、闭合、抬升、移动、下降、释放、稳定确认。不得长时间停顿、绕行、
抓空后继续、穿模、反复抓取或接受明显IK抖动。

按键：

| 按键 | 功能 |
| --- | --- |
| `W/S`、`A/D`、`R/F` | 末端平移 |
| 方向键、`Q/E` | 末端旋转 |
| 空格 | 切换夹爪 |
| `Z` | 取消当前尝试并以同一队列键重试 |
| `Enter` | 严格成功后确认保存 |
| `Backspace` | 严格成功后丢弃并以同一队列键重试 |
| `Esc` | 退出；未确认缓冲不会保存 |

首次有效操作才开始录制。采集为20 Hz、Viewer为60 Hz；达到400帧会自动
丢弃并重试当前键。

## 3. pilot蒙太奇与验收

生成蓝任务在左、黄任务在右，每列上方agentview、下方d435i_rgb的复核视频：

```powershell
python -m collector.v3.build_review_montages `
  --config configs\collect_mug_v3.yaml `
  --pilot
```

逐一观看 `review_montages/scene_4164.mp4` 和 `scene_2337.mp4`。确认没有抓空、
穿模、抖动或错误目标后，在下面文件中把对应状态改为 `pass`：

```text
smolvla-data/.smolvla_ur10e_mug_v3_collection/review_status.csv
```

然后执行自动验收和20 Hz确定性动作回放：

```powershell
python -m collector.v3.validate_dataset `
  --config configs\collect_mug_v3.yaml `
  --pilot
```

只有命令退出码为0且 `pilot_validation.json.status` 为 `pass`，正式采集入口才会
开放剩余36条。失败键会写入 `redo_keys.txt`。

## 4. 正式续采剩余36条

```powershell
python -m collector.v3.collect_matrix `
  --config configs\collect_mug_v3.yaml `
  --resume
```

任何中断都重复同一命令。程序只跳过契约、Parquet、双路视频和帧数均完整的键。

## 5. 全量蒙太奇、人工复核与正式验收

```powershell
python -m collector.v3.build_review_montages `
  --config configs\collect_mug_v3.yaml
```

观看全部20个scene视频，将 `review_status.csv` 的20行全部人工确认。存在问题时
不要标记为pass，只重采对应键：

```powershell
python -m collector.v3.collect_matrix `
  --config configs\collect_mug_v3.yaml `
  --resume `
  --redo-key "scene=4164|task=mug_on_blue|prompt=canonical"
```

重新生成并复核受影响scene的蒙太奇后，执行正式验收和最终物化：

```powershell
python -m collector.v3.validate_dataset `
  --config configs\collect_mug_v3.yaml `
  --finalize
```

PASS必须满足40条、20个共享scene、每任务20条、双路视频全部可解码、帧数一致、
夹爪闭合后释放、状态动作有限、4×5覆盖，以及40条从固定seed进行的确定性动作
回放严格成功。最终目录会包含：

```text
dataset_validation.json
episode_manifest.csv
redo_keys.txt
review_montages/
```

## 6. 回放最终数据

回放只读取数据集内容，不创建MuJoCo环境，也不执行动作：

```powershell
python -m collector.v3.replay `
  --root smolvla-data\smolvla_ur10e_mug_v3 `
  --episode-index 0
```

空格暂停，左右方向键逐帧，`Q`或`Esc`退出。

