# MIAA_SIM_v2

本项目是在 SAPIEN + Panda 仿真环境中实现的 VPG 与视觉伺服融合系统。系统使用腕部 RGB-D 相机感知工作空间，将 RGB-D 图像转换为 VPG 所需的 heightmap，由 VPG 模型决策 `push` 或 `grasp` 动作，再通过视觉伺服对目标点做局部 XY 修正，最后使用 mplib 在 SAPIEN 中执行推动或抓取 primitive。

当前保留两条主流程：

- VPG + 视觉伺服推理：`src.vpg_visual_servo_main`
- SAPIEN/Panda 下的 VPG 训练：`src.vpg_train_sapien`

根目录 `main.py` 是推理流程的兼容入口，等价于运行 `src.vpg_visual_servo_main`。

## 项目结构

```text
.
├── main.py                         # 兼容入口，运行 src.vpg_visual_servo_main
├── src/
│   ├── vpg_visual_servo_main.py    # VPG + 视觉伺服推理主流程
│   ├── vpg_train_sapien.py         # SAPIEN/Panda 下的 VPG 训练入口
│   ├── vpg_bridge/                 # RGB-D、heightmap、VPG 推理、伺服和 primitive 适配层
│   ├── vpg_training/               # VPG 训练环境封装
│   ├── config/                     # 机器人、物体、相机配置
│   ├── robot/                      # SAPIEN 场景控制和 mplib 运动规划
│   └── scale_urdf.py               # URDF mesh 缩放工具
├── visual-pushing-grasping-master/  # 原始 VPG 项目代码，供适配层复用
├── scripts/                        # 资产下载等辅助脚本
├── asset/                          # 仿真资产，本地使用，不纳入 Git 跟踪
├── logs/                           # 训练日志，本地生成
└── pic/                            # 报告和训练曲线图片
```

关键模块：

- `src/vpg_bridge/camera.py`：创建腕部 RGB-D 相机，读取 RGB、Depth 和内参，并处理 SAPIEN/VPG 相机坐标系转换。
- `src/vpg_bridge/heightmap.py`：将 RGB-D 图像投影为 VPG 所需的 color/depth heightmap。
- `src/vpg_bridge/policy.py`：封装原始 VPG `Trainer.forward()`，输出动作类型、目标点、动作角度和置信度。
- `src/vpg_bridge/servo.py`：根据目标点投影误差计算局部 XY 修正。
- `src/vpg_bridge/primitives.py`：实现 `grasp` 和 `push` primitive 的执行逻辑。
- `src/vpg_training/environment.py`：将 SAPIEN/Panda 封装为原 VPG 训练循环可用的环境。

## 准备工作

### 环境依赖

推荐运行环境：

- 操作系统：Ubuntu 22.04
- GPU：建议显存大于 8GB
- Python 环境管理：uv

在项目根目录执行：

```bash
uv sync
```

`uv sync` 会根据 `pyproject.toml` 和 `uv.lock` 创建/同步项目虚拟环境。核心依赖主要包括：

```text
sapien==3.0.0b1
mplib==0.2.1
numpy
opencv-python
scipy
torch
torchvision
matplotlib
```

### 仿真资产

仿真所需的 URDF、mesh 等大文件放在仓库根目录的 `asset/` 下。若克隆后没有 `asset/` 或资产不完整，可从 ModelScope 数据集拉取：

- 数据集地址：`arlenkang/MIAA_SIM`

需要先安装 Git LFS，然后在项目根目录执行：

```bash
.venv/bin/python scripts/download_assets_modelscope.py \
  --namespace arlenkang \
  --dataset MIAA_SIM \
  --local-dir .
```

默认会把远端数据集中的 `asset/` 同步到本地 `./asset/`。如果本地已有 `asset/`，脚本会整目录替换，请先自行备份。

## 推理使用方法

单轮推理：

```bash
.venv/bin/python main.py --iterations 1
```

或显式运行模块：

```bash
.venv/bin/python -m src.vpg_visual_servo_main --iterations 1
```

常用推理命令：

```bash
.venv/bin/python -m src.vpg_visual_servo_main \
  --iterations 10 \
  --num-objects 8 \
  --use-servo
```

关闭视觉伺服：

```bash
.venv/bin/python -m src.vpg_visual_servo_main \
  --iterations 10 \
  --no-servo
```

保存全局观察相机视频：

```bash
.venv/bin/python -m src.vpg_visual_servo_main \
  --iterations 10 \
  --save-video
```

使用训练得到的模型权重推理：

```bash
.venv/bin/python -m src.vpg_visual_servo_main \
  --snapshot-file logs/<session>/models/snapshot-backup.reinforcement.pth \
  --iterations 10 \
  --use-servo
```

只测试 VPG 策略本身：

```bash
.venv/bin/python -m src.vpg_visual_servo_main \
  --snapshot-file logs/<session>/models/snapshot-backup.reinforcement.pth \
  --iterations 10 \
  --no-servo
```

## 训练使用方法

检查 GPU：

```bash
.venv/bin/python - <<'PY'
import torch
print("cuda:", torch.cuda.is_available())
PY
```

CPU 可以用于烟雾测试，不适合正式训练。

### 训练命令

烟雾测试，确认 SAPIEN 场景、RGB-D、heightmap、日志和模型初始化可用：

```bash
.venv/bin/python -m src.vpg_train_sapien --max_iterations 5 --cpu
```

短训练，确认 push/grasp primitive、reward/label 日志和 backup snapshot 正常：

```bash
.venv/bin/python -m src.vpg_train_sapien \
  --max_iterations 200 \
  --push_rewards \
  --experience_replay \
  --explore_rate_decay
```

正式训练推荐参数：

```bash
.venv/bin/python -m src.vpg_train_sapien \
  --push_rewards \
  --experience_replay \
  --explore_rate_decay \
  --save_visualizations
```

限制正式训练轮数：

```bash
.venv/bin/python -m src.vpg_train_sapien \
  --max_iterations 5000 \
  --push_rewards \
  --experience_replay \
  --explore_rate_decay \
  --save_visualizations
```

默认每局生成 8 个 clutter 物体，可以修改：

```bash
.venv/bin/python -m src.vpg_train_sapien \
  --num-objects 10 \
  --push_rewards \
  --experience_replay \
  --explore_rate_decay
```

常用训练参数：

- `--push_rewards`：push 造成场景变化时给即时 reward `0.5`
- `--experience_replay`：启用原 VPG 的 prioritized experience replay
- `--explore_rate_decay`：探索率从 `0.5` 衰减到最低 `0.1`
- `--save_visualizations`：保存 push/grasp Q map 可视化

### 断点续训

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

### 训练输出

训练日志默认保存在：

```text
logs/<timestamp>/
├── data/
├── info/
├── models/
├── transitions/
└── visualizations/
```

关键 snapshot：

```text
logs/<session>/models/snapshot-backup.reinforcement.pth
logs/<session>/models/snapshot-000050.reinforcement.pth
logs/<session>/models/snapshot-000100.reinforcement.pth
...
```

其中：

- `snapshot-backup.reinforcement.pth`：每轮更新
- `snapshot-xxxxxx.reinforcement.pth`：每 50 轮保存一次

关键训练数据：

```text
logs/<session>/data/color-images/
logs/<session>/data/depth-images/
logs/<session>/data/color-heightmaps/
logs/<session>/data/depth-heightmaps/
```

关键训练记录：

```text
logs/<session>/transitions/executed-action.log.txt
logs/<session>/transitions/predicted-value.log.txt
logs/<session>/transitions/reward-value.log.txt
logs/<session>/transitions/label-value.log.txt
logs/<session>/transitions/is-exploit.log.txt
logs/<session>/transitions/use-heuristic.log.txt
logs/<session>/transitions/clearance.log.txt
```

## 推理过程与原理

融合后的系统采用分层控制：

```text
VPG 强化学习模型：
  根据 heightmap 决策动作类型、目标点和动作角度

视觉伺服：
  在执行前对 VPG 给出的目标点做局部修正

机械臂控制：
  使用 SAPIEN + mplib 执行 grasp 或 push primitive
```

整体流程：

```text
1. Panda 移动到固定观察位
2. 腕部 RGB-D 相机采集图像
3. RGB-D 图像转换为 VPG 所需 heightmap
4. VPG 输出 action、target_xyz、theta
5. 机械臂移动到 target_xyz 上方
6. 可选视觉伺服，对 target_xyz 做小范围 XY 修正
7. 根据 action 执行 grasp 或 push
8. 执行结束后回到观察位，进入下一轮
```

核心边界：

```text
VPG 决定“做什么、在哪里做、以什么角度做”
视觉伺服修正“是否对准”
机械臂控制执行“具体怎么做”
```

视觉伺服只允许局部修正：

```text
允许修改：
  target_x
  target_y

不修改：
  action 类型
  theta 动作角度
  push 方向
  push 长度
```

当 VPG 输出 `action == "grasp"`：

```text
1. 移动到 VPG 抓取点上方
2. 可选视觉伺服，修正抓取点附近的 XY 偏差
3. 保持 VPG 输出的抓取角度 theta
4. 打开夹爪
5. 下降到抓取高度
6. 闭合夹爪
7. 抬升
```

当 VPG 输出 `action == "push"`：

```text
1. 移动到 VPG 推动起点上方
2. 可选视觉伺服，修正推动起点附近的 XY 偏差
3. 保持 VPG 输出的推动方向 theta
4. 闭合夹爪
5. 下降到推动高度
6. 沿 theta 方向直线推动固定距离
7. 抬升
```

push 分支中的视觉伺服只修正推动起点，不改变推动方向。推动方向仍由 VPG 输出的 `theta` 决定。

## 训练过程与原理

训练入口是 `src.vpg_train_sapien`。该入口尽量保持原始 `visual-pushing-grasping-master` 的学习循环不变，只把执行环境从 UR5 + V-REP/CoppeliaSim 替换为 Panda + SAPIEN + mplib + 腕部 RGB-D 相机。

训练循环：

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

训练期间不使用视觉伺服。视觉伺服主要用于训练完成后的推理或后续真实机器人部署阶段的执行补偿。这样可以保持 VPG 动作像素和 reward/label 之间的因果关系。

训练入口复用原项目的：

```text
visual-pushing-grasping-master/trainer.py
visual-pushing-grasping-master/models.py
visual-pushing-grasping-master/logger.py
visual-pushing-grasping-master/utils.py
```

保持一致的关键设置包括：

- `method='reinforcement'`
- DenseNet121 ImageNet 预训练 backbone
- push/grasp heads 使用原代码 Kaiming 初始化
- `num_rotations = 16`
- `heightmap_resolution = 0.002`
- `future_reward_discount = 0.5`
- `explore_prob = 0.5`
- `SGD(lr=1e-4, momentum=0.9, weight_decay=2e-5)`
- Huber loss: `SmoothL1Loss(reduce=False)`
- 原 VPG 的 `change_detected`、`push_rewards`、`experience_replay`、`explore_rate_decay`
- 原 VPG 的日志目录和 snapshot 保存格式

从零训练时，原 VPG 会在 `models.py` 中使用：

```python
torchvision.models.densenet.densenet121(pretrained=True)
```

也就是说：

```text
DenseNet backbone: ImageNet 预训练
push/grasp heads: 随机初始化
VPG task snapshot: 不加载
```

## 常见问题

缺少 `torch` 或 `torchvision`：

```bash
.venv/bin/python -m pip install torch torchvision
```

训练很慢：

```text
VPG 原项目强依赖 GPU。CPU 适合烟雾测试，不适合正式训练。
```

reward 长期为 0 时，优先检查：

- 抓取高度 `grasp_z` 是否合适
- 方块位置是否在 Panda 舒适工作区
- depth heightmap 中物体高度是否明显
- 夹爪闭合参数是否合适

可以先查看：

```text
logs/<session>/transitions/reward-value.log.txt
logs/<session>/transitions/executed-action.log.txt
logs/<session>/data/depth-heightmaps/
logs/<session>/visualizations/
```

## 常用检查命令

查看推理入口参数：

```bash
.venv/bin/python main.py --help
```

查看训练入口参数：

```bash
.venv/bin/python -m src.vpg_train_sapien --help
```

检查 Python 文件语法：

```bash
python3 -m compileall main.py src
```
