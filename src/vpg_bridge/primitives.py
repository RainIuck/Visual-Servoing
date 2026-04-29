from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
from mplib import Pose

from .camera import DEFAULT_CAM_POSE_IN_HAND, RGBDFrame
from .heightmap import DEFAULT_WORKSPACE_LIMITS
from .servo import ServoResult, estimate_alignment_error


@dataclass
class PrimitiveConfig:
    safe_z: float = 0.50
    grasp_z: float = 0.125
    push_z: float = 0.10
    push_length: float = 0.10
    max_xy_correction: float = 0.02
    place_after_grasp: bool = False
    discard_origin: tuple[float, float, float] = (0.30, -0.34, 0.13)
    discard_spacing: float = 0.05
    placed_count: int = 0
    require_grasp_success_for_place: bool = True
    gripper_open_success_threshold: float = 0.005
    desired_gripper_pixel: tuple[float, float] = (320.0, 240.0)
    workspace_limits: np.ndarray = field(default_factory=lambda: DEFAULT_WORKSPACE_LIMITS.copy())
    cam_pose_in_hand: object = field(default_factory=lambda: DEFAULT_CAM_POSE_IN_HAND)


@dataclass
class ExecutionResult:
    action: str
    target_xyz: np.ndarray
    refined_xyz: np.ndarray
    theta: float
    servo: Optional[ServoResult]
    grasp_success: Optional[bool] = None


def execute_vpg_action(
    *,
    action: str,
    target_xyz: Sequence[float],
    theta: float,
    mp,
    hand_link,
    config: PrimitiveConfig,
    use_servo: bool,
    capture_rgbd: Callable[[], RGBDFrame],
) -> ExecutionResult:
    if action == "grasp":
        return execute_grasp(
            target_xyz=target_xyz,
            theta=theta,
            mp=mp,
            hand_link=hand_link,
            config=config,
            use_servo=use_servo,
            capture_rgbd=capture_rgbd,
        )
    if action == "push":
        return execute_push(
            target_xyz=target_xyz,
            theta=theta,
            mp=mp,
            hand_link=hand_link,
            config=config,
            use_servo=use_servo,
            capture_rgbd=capture_rgbd,
        )
    raise ValueError(f"Unknown VPG action: {action}")


def align_to_vpg_target(
    *,
    target_xyz: Sequence[float],
    theta: float,
    mp,
    hand_link,
    config: PrimitiveConfig,
    use_servo: bool,
    capture_rgbd: Callable[[], RGBDFrame],
) -> tuple[np.ndarray, list[float], Optional[ServoResult]]:
    target_xyz = np.asarray(target_xyz, dtype=np.float32)
    q = downward_gripper_quat(theta)
    mp.move_to_pose(Pose([float(target_xyz[0]), float(target_xyz[1]), config.safe_z], q))

    if not use_servo:
        return target_xyz.copy(), q, None

    frame = capture_rgbd()
    servo = estimate_alignment_error(
        target_xyz=target_xyz,
        current_ee_pose=hand_link.get_entity_pose(),
        cam_pose_in_hand=config.cam_pose_in_hand,
        intrinsics=frame.intrinsics,
        depth_img=frame.depth,
        desired_gripper_pixel=config.desired_gripper_pixel,
        max_xy_correction=config.max_xy_correction,
    )

    refined_xyz = target_xyz.copy()
    refined_xyz[:2] += servo.delta_xy
    if servo.ok and np.linalg.norm(servo.delta_xy) > 1e-5:
        mp.move_to_pose(Pose([float(refined_xyz[0]), float(refined_xyz[1]), config.safe_z], q))
    return refined_xyz, q, servo


def execute_grasp(
    *,
    target_xyz: Sequence[float],
    theta: float,
    mp,
    hand_link,
    config: PrimitiveConfig,
    use_servo: bool,
    capture_rgbd: Callable[[], RGBDFrame],
) -> ExecutionResult:
    refined_xyz, q, servo = align_to_vpg_target(
        target_xyz=target_xyz,
        theta=theta,
        mp=mp,
        hand_link=hand_link,
        config=config,
        use_servo=use_servo,
        capture_rgbd=capture_rgbd,
    )
    mp.open_gripper()
    mp.move_to_pose(Pose([float(refined_xyz[0]), float(refined_xyz[1]), config.grasp_z], q))
    mp.close_gripper()
    mp.move_to_pose(Pose([float(refined_xyz[0]), float(refined_xyz[1]), config.safe_z], q))
    grasp_success = estimate_gripper_blocked(mp, config.gripper_open_success_threshold)
    if config.place_after_grasp and (grasp_success or not config.require_grasp_success_for_place):
        drop_x, drop_y, drop_z = discard_pose(config)
        mp.move_to_pose(Pose([float(drop_x), float(drop_y), config.safe_z], q))
        mp.move_to_pose(Pose([float(drop_x), float(drop_y), float(drop_z)], q))
        mp.open_gripper()
        mp.move_to_pose(Pose([float(drop_x), float(drop_y), config.safe_z], q))
        config.placed_count += 1
    return ExecutionResult("grasp", np.asarray(target_xyz, dtype=np.float32), refined_xyz, theta, servo, grasp_success)


def execute_push(
    *,
    target_xyz: Sequence[float],
    theta: float,
    mp,
    hand_link,
    config: PrimitiveConfig,
    use_servo: bool,
    capture_rgbd: Callable[[], RGBDFrame],
) -> ExecutionResult:
    refined_xyz, q, servo = align_to_vpg_target(
        target_xyz=target_xyz,
        theta=theta,
        mp=mp,
        hand_link=hand_link,
        config=config,
        use_servo=use_servo,
        capture_rgbd=capture_rgbd,
    )

    x, y = float(refined_xyz[0]), float(refined_xyz[1])
    end_x = x + np.cos(theta) * config.push_length
    end_y = y + np.sin(theta) * config.push_length
    end_x = float(np.clip(end_x, config.workspace_limits[0][0], config.workspace_limits[0][1]))
    end_y = float(np.clip(end_y, config.workspace_limits[1][0], config.workspace_limits[1][1]))

    mp.close_gripper()
    mp.move_to_pose(Pose([x, y, config.push_z], q))
    mp.move_to_pose(Pose([end_x, end_y, config.push_z], q))
    mp.move_to_pose(Pose([end_x, end_y, config.safe_z], q))
    return ExecutionResult("push", np.asarray(target_xyz, dtype=np.float32), refined_xyz, theta, servo)


def downward_gripper_quat(theta: float) -> list[float]:
    base_down = [0.0, 1.0, 0.0, 0.0]
    qz = [float(np.cos(theta / 2.0)), 0.0, 0.0, float(np.sin(theta / 2.0))]
    return normalize_quat(quat_multiply(qz, base_down)).tolist()


def discard_pose(config: PrimitiveConfig) -> tuple[float, float, float]:
    x, y, z = config.discard_origin
    return float(x + config.discard_spacing * config.placed_count), float(y), float(z)


def estimate_gripper_blocked(mp, threshold: float = 0.005) -> bool:
    robot = getattr(mp.controller, "robot", None)
    if robot is None or not hasattr(robot, "get_qpos"):
        return True
    qpos = np.asarray(robot.get_qpos(), dtype=np.float32).reshape(-1)
    if qpos.size < 2:
        return True
    return bool(np.mean(np.abs(qpos[-2:])) > threshold)


def quat_multiply(q1: Sequence[float], q2: Sequence[float]) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.asarray(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float32,
    )


def normalize_quat(q: Sequence[float]) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    norm = float(np.linalg.norm(q))
    if norm <= 1e-8:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return q / norm
