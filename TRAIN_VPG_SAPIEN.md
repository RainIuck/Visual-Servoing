# 在 SAPIEN/Panda 中训练 VPG 模型

本文档说明如何在本项目的 SAPIEN + Panda 仿真环境中，从零训练 VPG 模型。训练方式尽量保持与原始 `visual-pushing-grasping-master` 项目一致，只把原来的 UR5 + V-REP 环境替换为本项目的 Panda + SAPIEN 环境。

## 1. 训练目标

训练入口：

```bash
src/vpg_train_sapien.py
```

训练后的模型权重可直接用于融合推理入口：

```bash
src/vpg_visual_servo_main.py
```

整体训练循环为：

```text
1. Panda 回到固定观察位
2. 腕部 RGB-D 相机采集图像
3. RGB-D 转换为 VPG heightmap
4. VPG 网络输出 push/grasp Q map
5. 根据原 VPG 逻辑选择动作、像素和角度
6. Panda 执行 push 或 grasp primitive
7. 再次观察场景
8. 根据 grasp_success/change_detected 计算 reward 和 label
9. 反向传播更新网络
10. 保存日志和 snapshot
```

注意：训练期间不使用视觉伺服。视觉伺服只用于训练完成后的推理/真实补偿阶段。

## 2. 与原 VPG 保持一致的部分

本训练入口复用原项目的：

```text
visual-pushing-grasping-master/trainer.py
visual-pushing-grasping-master/models.py
visual-pushing-grasping-master/logger.py
visual-pushing-grasping-master/utils.py
```

保持一致的训练逻辑包括：

- `method='reinforcement'`
- DenseNet121 ImageNet 预训练 backbone
- push/grasp heads 使用原代码 Kaiming 初始化
- `num_rotations = 16`
- `heightmap_resolution = 0.002`
- `future_reward_discount = 0.5`
- `explore_prob = 0.5`
- `SGD(lr=1e-4, momentum=0.9, weight_decay=2e-5)`
- Huber loss: `SmoothL1Loss(reduce=False)`
- 原 VPG 的 `change_detected` 判定逻辑
- 原 VPG 的 `push_rewards`、`experience_replay`、`explore_rate_decay`
- 原 VPG 的日志目录和 snapshot 保存格式

不同的部分只有仿真和执行环境：

```text
原项目：UR5 + RG2 + V-REP/CoppeliaSim
本项目：Panda + SAPIEN + mplib + 腕部 RGB-D 相机
```

## 3. 环境依赖

项目已有依赖：

```bash
pip install sapien==3.0.0b1 mplib==0.2.1
```

VPG 训练还需要：

```bash
pip install numpy scipy opencv-python matplotlib torch torchvision
```

推荐使用项目虚拟环境：

```bash
.venv/bin/python -m pip install numpy scipy opencv-python matplotlib torch torchvision
```

如果使用 GPU，请按你的 CUDA 版本安装对应的 PyTorch。安装完成后检查：

```bash
.venv/bin/python - <<'PY'
import torch
import torchvision
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("cuda:", torch.cuda.is_available())
PY
```

## 4. ImageNet DenseNet121 权重

原 VPG 不是全随机训练。它在 `models.py` 中使用：

```python
torchvision.models.densenet.densenet121(pretrained=True)
```

也就是说：

```text
DenseNet backbone: ImageNet 预训练
push/grasp heads: 随机初始化
VPG task snapshot: 不加载
```

因此训练前需要确保 torchvision 能拿到 DenseNet121 ImageNet 权重。

如果机器可以联网，第一次运行训练时 torchvision 会自动下载。

如果机器不能联网，需要提前把 DenseNet121 权重放入 torch cache。常见 cache 位置为：

```text
~/.cache/torch/hub/checkpoints/
```

具体文件名取决于 torchvision 版本，通常类似：

```text
densenet121-a639ec97.pth
```

如果缺少该权重，训练会在初始化模型时失败。这是为了保持与原 VPG 完全一致的训练方式，不回退到全随机 backbone。

## 5. 快速烟雾测试

先跑一个很短的测试，确认 SAPIEN 场景、RGB-D、heightmap、日志和模型初始化都能工作：

```bash
.venv/bin/python -m src.vpg_train_sapien \
  --max_iterations 5 \
  --cpu
```

如果有 GPU，建议去掉 `--cpu`：

```bash
.venv/bin/python -m src.vpg_train_sapien \
  --max_iterations 5
```

成功时会看到类似信息：

```text
Creating data logging session: logs/YYYY-MM-DD.HH:MM:SS
Training iteration: 0
Primitive confidence scores: ...
Action: grasp/push at (...)
Executing: ...
Push successful: ...
Grasp successful: ...
```

并生成：

```text
logs/<session>/
├── data/
├── info/
├── models/
├── transitions/
└── visualizations/
```

## 6. 正式训练命令

推荐使用与原 VPG 论文常用设置一致的训练参数：

```bash
.venv/bin/python -m src.vpg_train_sapien \
  --push_rewards \
  --experience_replay \
  --explore_rate_decay \
  --save_visualizations
```

含义：

- `--push_rewards`：push 造成场景变化时给即时 reward `0.5`
- `--experience_replay`：启用原 VPG 的 prioritized experience replay
- `--explore_rate_decay`：探索率从 `0.5` 衰减到最低 `0.1`
- `--save_visualizations`：保存 push/grasp Q map 可视化

如果只想限制训练轮数：

```bash
.venv/bin/python -m src.vpg_train_sapien \
  --max_iterations 5000 \
  --push_rewards \
  --experience_replay \
  --explore_rate_decay \
  --save_visualizations
```

默认每局生成 8 个方块 clutter。可以修改：

```bash
.venv/bin/python -m src.vpg_train_sapien \
  --num-objects 10 \
  --push_rewards \
  --experience_replay \
  --explore_rate_decay
```

## 7. 训练日志和模型权重

训练日志默认保存在：

```text
logs/<timestamp>/
```

关键文件：

```text
logs/<session>/models/snapshot-backup.reinforcement.pth
logs/<session>/models/snapshot-000050.reinforcement.pth
logs/<session>/models/snapshot-000100.reinforcement.pth
...
```

其中：

- `snapshot-backup.reinforcement.pth`：每轮更新
- `snapshot-xxxxxx.reinforcement.pth`：每 50 轮保存一次

训练数据：

```text
logs/<session>/data/color-images/
logs/<session>/data/depth-images/
logs/<session>/data/color-heightmaps/
logs/<session>/data/depth-heightmaps/
```

训练记录：

```text
logs/<session>/transitions/executed-action.log.txt
logs/<session>/transitions/predicted-value.log.txt
logs/<session>/transitions/reward-value.log.txt
logs/<session>/transitions/label-value.log.txt
logs/<session>/transitions/is-exploit.log.txt
logs/<session>/transitions/use-heuristic.log.txt
logs/<session>/transitions/clearance.log.txt
```

## 8. 断点续训

如果训练中断，可以用 backup snapshot 继续：

```bash
.venv/bin/python -m src.vpg_train_sapien \
  --continue_logging \
  --logging_directory logs/<session> \
  --load_snapshot \
  --snapshot_file logs/<session>/models/snapshot-backup.reinforcement.pth \
  --push_rewards \
  --experience_replay \
  --explore_rate_decay \
  --save_visualizations
```

注意：

- `--logging_directory` 必须指向已有 session 目录。
- 继续训练时要保留与原训练一致的 reward/replay 参数。
- 如果原训练用了 `--push_rewards`，续训也要加上。

## 9. 使用训练好的模型推理

训练完成后，用融合推理入口加载 snapshot：

```bash
.venv/bin/python -m src.vpg_visual_servo_main \
  --snapshot-file logs/<session>/models/snapshot-backup.reinforcement.pth \
  --iterations 10 \
  --use-servo
```

如果想先关闭视觉伺服，只测试 VPG 策略本身：

```bash
.venv/bin/python -m src.vpg_visual_servo_main \
  --snapshot-file logs/<session>/models/snapshot-backup.reinforcement.pth \
  --iterations 10 \
  --no-servo
```

## 10. 建议训练阶段

建议按以下顺序推进：

### 阶段 1：烟雾测试

```bash
.venv/bin/python -m src.vpg_train_sapien \
  --max_iterations 5 \
  --cpu
```

目标：

- 程序能启动
- RGB-D 能采集
- heightmap 能生成
- 日志能写入

### 阶段 2：短训练

```bash
.venv/bin/python -m src.vpg_train_sapien \
  --max_iterations 200 \
  --push_rewards \
  --experience_replay \
  --explore_rate_decay
```

目标：

- push/grasp primitive 都能执行
- reward/label 日志正常增长
- `snapshot-backup.reinforcement.pth` 能生成

### 阶段 3：正式训练

```bash
.venv/bin/python -m src.vpg_train_sapien \
  --max_iterations 5000 \
  --push_rewards \
  --experience_replay \
  --explore_rate_decay \
  --save_visualizations
```

目标：

- grasp reward 不长期为 0
- 策略逐渐优于随机动作
- snapshot 可用于推理入口

## 11. 常见问题

### 缺少 torch

报错：

```text
Missing dependency: torch
```

解决：

```bash
.venv/bin/python -m pip install torch torchvision
```

### DenseNet121 权重下载失败

如果 torchvision 无法下载 ImageNet 权重，需要手动准备 DenseNet121 权重到：

```text
~/.cache/torch/hub/checkpoints/
```

本项目不会自动改为全随机初始化，因为训练目标是保持与原 VPG 一致。

### 训练很慢

VPG 原项目强烈依赖 GPU。CPU 可以用于烟雾测试，但不适合正式训练。

检查 GPU：

```bash
.venv/bin/python - <<'PY'
import torch
print(torch.cuda.is_available())
PY
```

### reward 长期为 0

可能原因：

- 抓取高度 `grasp_z` 不合适
- 方块位置不在 Panda 舒适工作区
- heightmap 中物体高度不明显
- 夹爪闭合参数不合适

建议先用短训练观察：

```text
transitions/reward-value.log.txt
transitions/executed-action.log.txt
data/depth-heightmaps/
visualizations/
```

### 训练期间不要打开视觉伺服

训练入口默认不使用视觉伺服。原因是 VPG 的 reward/label 应该对应网络选择的原始目标点。如果训练时伺服改变执行点，会破坏动作像素和 reward 之间的因果关系。
