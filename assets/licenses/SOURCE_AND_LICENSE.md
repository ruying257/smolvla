# MuJoCo 场景资源来源与许可证

## 来源

本目录对应的模型资源来自本机ACT参考项目：

`F:\桌面\code_learn\mujoco-act-robotics\mode`

主场景 `assets/mujoco/scene.xml` 以其中的 `demo_scene.xml` 为结构参考，保留机器人、桌面、相机、地面、天空盒、灯光、观察参数和世界坐标轴；删除杯子与盘子，并新增SmolVLA任务使用的红绿积木和蓝黄放置区域。

并行场景 `assets/mujoco/mug_scene.xml` 复用相同的机器人、桌面、相机和背景，并从ACT参考项目完整复制 `mode/mug_5/`。复制的源文件保持逐字节一致；新增的 `mug_5/model_smolvla.xml` 只为直接展示提供安全的桌面默认位置，其视觉网格、缩放、纹理、密度、摩擦和碰撞参数均来自 `model_new.xml`。实验纹理 `mug_5/visual/image0_green_white.png` 由原始 `image0.png` 仅转换红色区域的色相得到，白色区域、黑色Logo、分辨率和纹理布局保持不变。

为支持包含中文字符的Windows项目路径，`ur10e_with_2f85_d435i.xml` 仅修正资源定位方式。机器人、夹爪、相机和桌面的位姿、尺寸、质量、惯量及控制参数未因此改变。

## 许可证

- Universal Robots UR10e模型：BSD 3-Clause，完整文本见 `universal_robots_ur10e_LICENSE`。
- Robotiq 2F-85模型：BSD 2-Clause，完整文本见 `robotiq_2f85_LICENSE`。
- Intel RealSense D435i模型：Apache License 2.0，完整文本见 `realsense_d435i_LICENSE`。
- `mug_5`杯子模型：来源于ACT参考项目；其许可范围以ACT仓库及模型原始来源声明为准。
- ACT项目自身及桌面、木纹资源的许可范围应以ACT参考仓库中的声明和原始资源来源为准；本项目不扩大其授权范围。

逐文件来源摘要由 `scripts/generate_asset_manifest.py` 生成到同目录的 `asset_manifest.json`。清单中的 `copied` 表示文件与ACT源文件SHA-256一致，`modified` 表示为本项目兼容或任务场景所作的修改。
