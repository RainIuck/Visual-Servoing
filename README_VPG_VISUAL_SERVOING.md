# VPG 与视觉伺服融合后的控制逻辑

融合后的系统采用分层控制：

```text
VPG 强化学习模型：
  根据 heightmap 决策动作类型、目标点和动作角度

视觉伺服：
  在执行前对 VPG 给出的目标点做局部修正

本项目机械臂控制：
  使用 SAPIEN + mplib 执行 grasp 或 push primitive
```

## 1. 总体流程

```text
1. 机械臂移动到固定观察位
2. 腕部 RGB-D 相机采集图像
3. RGB-D 图像转换为 VPG 所需 heightmap
4. VPG 输出：
   - action: grasp 或 push
   - target_xyz: 动作目标点
   - theta: 动作角度
5. 机械臂移动到 target_xyz 上方
6. 可选视觉伺服，对 target_xyz 做小范围修正
7. 根据 action 执行不同控制逻辑
8. 执行结束后回到观察位，进入下一轮
```

核心思想：

```text
VPG 决定“做什么、在哪里做、以什么角度做”
视觉伺服修正“是否对准”
机械臂控制执行“具体怎么做”
```

## 2. 公共对准阶段

`grasp` 和 `push` 都可以复用同一个目标点对准阶段。

```python
def align_to_vpg_target(target_xyz, theta, action, use_servo=True):
    move_above_target(target_xyz, theta)

    if not use_servo:
        return target_xyz, theta

    refined_xyz = visual_servo_refine_target(
        target_xyz=target_xyz,
        theta=theta,
        action=action,
        max_xy_correction=0.02,
    )

    return refined_xyz, theta
```

视觉伺服只做局部修正：

```text
允许修改：
  target_x
  target_y

不建议修改：
  action 类型
  theta 动作角度
  push 方向
  push 长度
```

## 3. Grasp 控制逻辑

当 VPG 输出 `action == grasp`：

```text
1. 移动到 VPG 抓取点上方
2. 可选视觉伺服，修正抓取点附近的 XY 偏差
3. 保持 VPG 输出的抓取角度 theta
4. 打开夹爪
5. 下降到抓取高度
6. 闭合夹爪
7. 抬升
```

伪代码：

```python
def execute_grasp(target_xyz, theta):
    refined_xyz, theta = align_to_vpg_target(
        target_xyz=target_xyz,
        theta=theta,
        action="grasp",
        use_servo=True,
    )

    q = downward_gripper_quat(theta)

    mp.open_gripper()
    mp.move_to_pose(Pose([refined_xyz[0], refined_xyz[1], grasp_z], q))
    mp.close_gripper()
    mp.move_to_pose(Pose([refined_xyz[0], refined_xyz[1], safe_z], q))
```

注意：视觉伺服不能替代 VPG 重新选择抓取点。它只负责在 VPG 抓取点附近做小范围误差补偿。

## 4. Push 控制逻辑

当 VPG 输出 `action == push`：

```text
1. 移动到 VPG 推动起点上方
2. 可选视觉伺服，修正推动起点附近的 XY 偏差
3. 保持 VPG 输出的推动方向 theta
4. 闭合夹爪
5. 下降到推动高度
6. 沿 theta 方向直线推动固定距离
7. 抬升
```

伪代码：

```python
def execute_push(target_xyz, theta):
    refined_xyz, theta = align_to_vpg_target(
        target_xyz=target_xyz,
        theta=theta,
        action="push",
        use_servo=True,
    )

    x, y, z = refined_xyz
    q = downward_gripper_quat(theta)

    push_len = 0.10
    end_x = x + np.cos(theta) * push_len
    end_y = y + np.sin(theta) * push_len

    end_x = np.clip(end_x, workspace_limits[0][0], workspace_limits[0][1])
    end_y = np.clip(end_y, workspace_limits[1][0], workspace_limits[1][1])

    mp.close_gripper()
    mp.move_to_pose(Pose([x, y, push_z], q))
    mp.move_to_pose(Pose([end_x, end_y, push_z], q))
    mp.move_to_pose(Pose([end_x, end_y, safe_z], q))
```

注意：push 分支中的视觉伺服只修正推动起点，不改变推动方向。推动方向仍由 VPG 输出的 `theta` 决定。

## 5. 主控制入口

```python
def execute_vpg_action(action, target_xyz, theta):
    if action == "grasp":
        execute_grasp(target_xyz, theta)
    elif action == "push":
        execute_push(target_xyz, theta)
    else:
        raise ValueError(f"Unknown action: {action}")
```

完整循环：

```python
while True:
    move_to_observation_pose()

    color_img, depth_img = capture_rgbd()
    color_heightmap, depth_heightmap = build_heightmap(color_img, depth_img)

    action, target_xyz, theta = vpg_policy.predict(
        color_heightmap,
        depth_heightmap,
    )

    execute_vpg_action(action, target_xyz, theta)
```

