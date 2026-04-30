# VPG 与 SAPIEN 视觉伺服融合实施计划

## Summary

第一版实现 **VPG 推理闭环**，不迁移训练逻辑。目标流程为：

`Panda 回固定观察位 -> 腕部 RGB-D 相机采集 -> 生成 VPG heightmap -> VPG 输出 action/target_xyz/theta -> 机械臂移动到目标点上方 -> RGB-D 视觉伺服检测目标点投影误差 -> 必要时小范围闭环修正 -> 执行 grasp 或 push -> 回观察位循环`

视觉伺服统一按真实机器人可用方式实现：使用当前末端位姿、手眼外参、腕部 RGB-D、相机内参，把 VPG 目标点投影到当前相机图像中，与夹爪作用点标定像素比较。在理想仿真中，误差应接近 `0`，正常不修正。

## Key Changes

- 单红块 RGB/HSV 普通视觉伺服 demo 已移除。
- 融合主入口为 `src/vpg_visual_servo_main.py`。
- 新增 `src/vpg_bridge/`：
  - `camera.py`：创建/读取腕部 RGB-D 相机，提供 RGB、depth、内参。
  - `heightmap.py`：将 RGB-D 转为 VPG 所需 heightmap。
  - `policy.py`：封装 VPG `Trainer.forward()`，输出 `action, target_xyz, theta, confidence`。
  - `servo.py`：实现统一 RGB-D 投影闭环伺服。
  - `primitives.py`：实现 `align_to_vpg_target()`、`execute_grasp()`、`execute_push()`。
- VPG 原项目保持只读，通过适配层复用 `trainer.py`、`models.py`、`utils.py`。

## Camera And Servo Design

- 算法只使用 **1 个腕部 RGB-D 相机**。
- 相机挂载到 `panda_hand`，初始采用：
  ```python
  cam_pose_in_hand = sapien.Pose([0.05, 0.0, 0.04], [0.7071, 0, -0.7071, 0])
  ```
- 每次 VPG 推理前，机械臂移动到固定观察位：
  ```python
  Pose([0.40, 0.10, 0.50], [0, 1, 0, 0])
  ```
- `capture_rgbd()` 返回：
  ```python
  color_img  # H x W x 3, RGB uint8
  depth_img  # H x W, meters float32
  intrinsics # 3 x 3 camera matrix
  ```
- 视觉伺服统一数据流：
  ```text
  current_ee_pose + cam_pose_in_hand -> camera_pose_world
  target_xyz -> target_camera_xyz -> target_pixel
  error_pixel = target_pixel - desired_gripper_pixel
  error_pixel + depth + intrinsics -> servo_delta_camera
  servo_delta_camera -> servo_delta_world_xy
  ```
- `desired_gripper_pixel` 是夹爪作用点标定像素，第一版默认 `[320, 240]`，作为参数保留。
- 视觉伺服限制：
  - 只输出 `servo_delta_xy`。
  - 最大修正 `0.02m`。
  - 不改变 VPG 的 `action`、`theta`、push 方向、push 长度。
  - 不寻找物体中心，不用 ROI 质心替代 VPG 目标点。
  - 仿真与真实使用同一算法，不使用仿真真值误差作为控制输入。

## Implementation Changes

- Heightmap：
  - 默认 workspace：
    ```python
    [[0.176, 0.624], [-0.224, 0.224], [0.0, 0.4]]
    ```
  - 默认 `heightmap_resolution = 0.002`。
  - 输出 `224 x 224` 的 `color_heightmap` 和 `depth_heightmap`。
- VPG 推理：
  - 加载 snapshot：
    `visual-pushing-grasping-master/downloads/vpg-original-sim-pretrained-10-obj.pth`
  - 比较 push/grasp 最大预测值，得到动作类型。
  - 将最佳像素和旋转索引转换为 Panda workspace 下的 `target_xyz` 与 `theta`。
- 视觉伺服：
  - 机械臂先移动到 VPG 输出目标点上方。
  - 读取当前末端位姿和腕部 RGB-D。
  - 投影 `target_xyz` 到当前图像，计算 `target_pixel - desired_gripper_pixel`。
  - 根据目标深度与内参将像素误差转换成小范围 XY 修正。
  - 若投影越界、深度无效或修正超过阈值，记录失败并返回零修正。
- Primitive 执行：
  - `grasp`：对准目标点 -> 可选伺服补偿 -> 张爪 -> 下探 -> 闭爪 -> 抬升。
  - `push`：对准推动起点 -> 可选伺服补偿 -> 闭爪 -> 下探 -> 沿 `theta` 推动 `0.10m` -> 抬升。
  - push 终点裁剪到 workspace 内。
- 场景：
  - 默认生成 `8` 个方块形成 clutter。
  - 方块随机分布在 workspace 内，避免严重初始重叠。
  - 每轮动作结束后回固定观察位重新采集 RGB-D。

## Test Plan

- 相机测试：
  - 验证 RGB 和 depth 都能读取。
  - 验证 depth 单位为米，且物体区域深度有效。
  - 验证内参矩阵可用于投影。
- Heightmap 测试：
  - 验证 heightmap 尺寸为 `224 x 224`。
  - 验证方块区域在 depth heightmap 中有合理高度。
- VPG 推理测试：
  - snapshot 加载成功。
  - 单轮输出 `action in {"push", "grasp"}`。
  - `target_xyz` 落在 workspace 内。
- 视觉伺服测试：
  - 理想仿真无扰动时，`target_pixel` 接近 `desired_gripper_pixel`，`servo_delta_xy` 接近 `[0, 0]`。
  - 人为设置小偏差时，像素误差方向与 `servo_delta_xy` 方向一致，且不超过 `0.02m`。
  - 投影越界或深度无效时，不执行错误修正。
- 集成测试：
  - `--iterations 1 --no-servo` 跑通 VPG 到 primitive。
  - `--iterations 1 --use-servo` 跑通 RGB-D 投影伺服路径。
  - `--iterations 10` 验证循环稳定返回观察位。

## Assumptions

- 第一版只实现推理闭环，不做训练、reward、experience replay、logger 迁移。
- 视觉伺服只补偿执行对准误差，不修改 VPG 决策语义。
- 在理想仿真环境中，视觉伺服默认应不产生实际修正。
- 如果 VPG snapshot 不存在，程序提示下载权重并退出。
