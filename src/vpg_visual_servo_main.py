from __future__ import annotations

import argparse
import copy
import random
import sys
from pathlib import Path

import numpy as np
import sapien.core as sapien
from mplib import Pose

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config.object_varant import (
    block_config,
    block_green_config,
    block_red_config,
    block_yellow_config,
)
from src.config.robot_varant import panda_config
from src.robot.controller import Controller
from src.robot.motion_planning import MotionPlanning
from src.vpg_bridge.camera import (
    DEFAULT_CAM_POSE_IN_HAND,
    capture_rgbd,
    create_wrist_rgbd_camera,
    get_camera_pose_world,
    vpg_camera_pose_from_sapien,
)
from src.vpg_bridge.heightmap import (
    DEFAULT_HEIGHTMAP_RESOLUTION,
    DEFAULT_WORKSPACE_LIMITS,
    build_heightmap,
)
from src.vpg_bridge.policy import VPGPolicy
from src.vpg_bridge.primitives import PrimitiveConfig, execute_vpg_action


PROJECT_DIR = Path(__file__).resolve().parents[1]
VPG_DIR = PROJECT_DIR / "visual-pushing-grasping-master"
DEFAULT_SNAPSHOT = VPG_DIR / "downloads/vpg-original-sim-pretrained-10-obj.pth"
OBSERVATION_POSE = Pose([0.40, 0.10, 0.50], [0, 1, 0, 0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VPG inference with SAPIEN Panda visual servoing.")
    parser.add_argument("--snapshot-file", default=str(DEFAULT_SNAPSHOT), help="Path to a VPG snapshot .pth file.")
    parser.add_argument("--iterations", type=int, default=1, help="Number of VPG control iterations.")
    parser.add_argument("--num-objects", type=int, default=8, help="Number of block objects to spawn.")
    parser.add_argument("--cpu", action="store_true", help="Force VPG inference to run on CPU.")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for clutter placement.")
    parser.add_argument("--use-servo", dest="use_servo", action="store_true", default=True)
    parser.add_argument("--no-servo", dest="use_servo", action="store_false")
    parser.add_argument("--keep-viewer", action="store_true", help="Keep the SAPIEN viewer open after the loop.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    snapshot_path = Path(args.snapshot_file).resolve()
    if not snapshot_path.exists():
        raise SystemExit(
            f"VPG snapshot not found: {snapshot_path}\n"
            "Download weights first, for example:\n"
            "  cd visual-pushing-grasping-master/downloads && ./download-weights.sh"
        )

    policy = VPGPolicy(
        vpg_dir=VPG_DIR,
        snapshot_file=snapshot_path,
        workspace_limits=DEFAULT_WORKSPACE_LIMITS,
        heightmap_resolution=DEFAULT_HEIGHTMAP_RESOLUTION,
        force_cpu=args.cpu,
    )

    controller = Controller()
    robot = setup_robot(controller)
    camera, hand_link, intrinsics = create_wrist_rgbd_camera(
        controller.scene,
        robot,
        pose_in_hand=DEFAULT_CAM_POSE_IN_HAND,
    )
    spawn_clutter(controller, args.num_objects)
    step_scene(controller, robot, steps=80)

    mp = MotionPlanning(panda_config, controller)
    primitive_config = PrimitiveConfig(
        workspace_limits=DEFAULT_WORKSPACE_LIMITS.copy(),
        cam_pose_in_hand=DEFAULT_CAM_POSE_IN_HAND,
    )
    def capture_current_frame():
        return capture_rgbd(camera, scene=controller.scene, intrinsics=intrinsics)

    try:
        for iteration in range(args.iterations):
            print(f"\n[VPG] Iteration {iteration + 1}/{args.iterations}")
            mp.move_to_pose(OBSERVATION_POSE)

            frame = capture_current_frame()
            camera_pose_world = get_camera_pose_world(hand_link.get_entity_pose(), DEFAULT_CAM_POSE_IN_HAND)
            vpg_camera_pose = vpg_camera_pose_from_sapien(camera_pose_world)
            color_heightmap, depth_heightmap = build_heightmap(
                frame.color,
                frame.depth,
                frame.intrinsics,
                vpg_camera_pose,
                vpg_dir=VPG_DIR,
                workspace_limits=DEFAULT_WORKSPACE_LIMITS,
                heightmap_resolution=DEFAULT_HEIGHTMAP_RESOLUTION,
            )

            print(
                "[VPG] RGB-D "
                f"color={frame.color.shape} depth={frame.depth.shape} "
                f"heightmap={color_heightmap.shape[:2]}"
            )
            vpg_action = policy.predict(color_heightmap, depth_heightmap)
            print(
                "[VPG] action={action} confidence={confidence:.4f} "
                "push={push:.4f} grasp={grasp:.4f} target={target} theta={theta:.3f} pix={pix}".format(
                    action=vpg_action.action,
                    confidence=vpg_action.confidence,
                    push=vpg_action.push_confidence,
                    grasp=vpg_action.grasp_confidence,
                    target=np.round(vpg_action.target_xyz, 4).tolist(),
                    theta=vpg_action.theta,
                    pix=vpg_action.best_pix_ind,
                )
            )

            result = execute_vpg_action(
                action=vpg_action.action,
                target_xyz=vpg_action.target_xyz,
                theta=vpg_action.theta,
                mp=mp,
                hand_link=hand_link,
                config=primitive_config,
                use_servo=args.use_servo,
                capture_rgbd=capture_current_frame,
            )
            print_execution_result(result)
    finally:
        controller.out.release()

    if args.keep_viewer:
        controller.visualize(robot)


def setup_robot(controller: Controller):
    robot_config = copy.deepcopy(panda_config)
    robot_config["position"] = [0.0, 0.0, 0.0]
    robot = controller.add_robot(robot_config)
    safe_home_qpos = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04]
    robot.set_qpos(safe_home_qpos)
    for i, joint in enumerate(robot.active_joints):
        if i < len(safe_home_qpos):
            joint.set_drive_target(safe_home_qpos[i])
    step_scene(controller, robot, steps=50)
    return robot


def spawn_clutter(controller: Controller, num_objects: int) -> None:
    configs = [block_red_config, block_green_config, block_yellow_config, block_config]
    positions = sample_clutter_positions(num_objects)
    for idx, position in enumerate(positions):
        cfg = copy.deepcopy(configs[idx % len(configs)])
        cfg["object_name"] = f"{cfg.get('object_name', 'block')}_{idx:02d}"
        cfg["position"] = [float(position[0]), float(position[1]), 0.02]
        controller.add_object(cfg)
    print(f"[Scene] Spawned {len(positions)} clutter blocks.")


def sample_clutter_positions(num_objects: int) -> list[tuple[float, float]]:
    block_size = 0.04
    spacing = block_size + 0.003
    cols = int(np.ceil(np.sqrt(num_objects)))
    rows = int(np.ceil(num_objects / cols))
    center_x = random.uniform(0.37, 0.43)
    center_y = random.uniform(-0.04, 0.04)

    grid = []
    for row in range(rows):
        for col in range(cols):
            x = (col - (cols - 1) / 2.0) * spacing
            y = (row - (rows - 1) / 2.0) * spacing
            grid.append((x, y))
    random.shuffle(grid)

    positions = []
    margin = block_size / 2.0
    for dx, dy in grid[:num_objects]:
        x = center_x + dx + random.uniform(-0.002, 0.002)
        y = center_y + dy + random.uniform(-0.002, 0.002)
        x = np.clip(x, DEFAULT_WORKSPACE_LIMITS[0][0] + margin, DEFAULT_WORKSPACE_LIMITS[0][1] - margin)
        y = np.clip(y, DEFAULT_WORKSPACE_LIMITS[1][0] + margin, DEFAULT_WORKSPACE_LIMITS[1][1] - margin)
        positions.append((float(x), float(y)))
    return positions


def step_scene(controller: Controller, robot, *, steps: int) -> None:
    for i in range(steps):
        qf = robot.compute_passive_force(gravity=True, coriolis_and_centrifugal=True)
        robot.set_qf(qf)
        controller.scene.step()
        if i % 4 == 0 and controller.viewer:
            controller.scene.update_render()
            controller.viewer.render()


def print_execution_result(result) -> None:
    if result.servo is None:
        print("[Servo] disabled")
    else:
        print(
            "[Servo] status={status} delta_xy={delta} target_pixel={target_pixel} "
            "pixel_error={pixel_error}".format(
                status=result.servo.status,
                delta=np.round(result.servo.delta_xy, 5).tolist(),
                target_pixel=(
                    None if result.servo.target_pixel is None else np.round(result.servo.target_pixel, 2).tolist()
                ),
                pixel_error=(
                    None if result.servo.pixel_error is None else np.round(result.servo.pixel_error, 2).tolist()
                ),
            )
        )
    print(
        "[Exec] action={action} target={target} refined={refined} theta={theta:.3f}".format(
            action=result.action,
            target=np.round(result.target_xyz, 4).tolist(),
            refined=np.round(result.refined_xyz, 4).tolist(),
            theta=result.theta,
        )
    )


if __name__ == "__main__":
    main()
