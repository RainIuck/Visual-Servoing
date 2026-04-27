from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Sequence

import numpy as np
from mplib import Pose

from src.config.object_varant import block_config, block_green_config, block_red_config, block_yellow_config
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
from src.vpg_bridge.heightmap import DEFAULT_HEIGHTMAP_RESOLUTION, DEFAULT_WORKSPACE_LIMITS, build_heightmap
from src.vpg_bridge.primitives import PrimitiveConfig, execute_grasp, execute_push


PROJECT_DIR = Path(__file__).resolve().parents[2]
VPG_DIR = PROJECT_DIR / "visual-pushing-grasping-master"
OBSERVATION_POSE = Pose([0.40, 0.10, 0.50], [0, 1, 0, 0])


class SapienVPGEnvironment:
    """SAPIEN/Panda drop-in environment for the original VPG training loop."""

    def __init__(
        self,
        *,
        num_obj: int = 8,
        workspace_limits: np.ndarray = DEFAULT_WORKSPACE_LIMITS,
        heightmap_resolution: float = DEFAULT_HEIGHTMAP_RESOLUTION,
        random_seed: int = 1234,
        save_video: bool = False,
    ) -> None:
        self.num_obj = int(num_obj)
        self.workspace_limits = np.asarray(workspace_limits, dtype=np.float32)
        self.heightmap_resolution = float(heightmap_resolution)
        self.random = random.Random(random_seed)
        self.controller = Controller()
        self.robot = self._setup_robot()
        self.camera, self.hand_link, self.cam_intrinsics = create_wrist_rgbd_camera(
            self.controller.scene,
            self.robot,
            pose_in_hand=DEFAULT_CAM_POSE_IN_HAND,
        )
        self.cam_pose = np.eye(4, dtype=np.float32)
        self.cam_depth_scale = 1.0
        self._update_camera_pose()
        self.mp = MotionPlanning(panda_config, self.controller)
        self.primitive_config = PrimitiveConfig(
            workspace_limits=self.workspace_limits.copy(),
            cam_pose_in_hand=DEFAULT_CAM_POSE_IN_HAND,
        )
        self.object_handles = []
        self.object_configs = [block_red_config, block_green_config, block_yellow_config, block_config]
        self.discard_origin = np.asarray([0.75, 0.45, 0.08], dtype=np.float32)
        self.save_video = save_video
        self.add_objects()

    def close(self) -> None:
        if hasattr(self.controller, "out") and self.controller.out:
            self.controller.out.release()

    def move_to_observation(self) -> None:
        self.mp.move_to_pose(OBSERVATION_POSE)
        self._update_camera_pose()

    def get_camera_data(self):
        self.move_to_observation()
        frame = capture_rgbd(self.camera, scene=self.controller.scene, intrinsics=self.cam_intrinsics)
        self._update_camera_pose()
        return frame.color, frame.depth

    def get_heightmaps(self, color_img: np.ndarray, depth_img: np.ndarray):
        return build_heightmap(
            color_img,
            depth_img,
            self.cam_intrinsics,
            self.cam_pose,
            vpg_dir=VPG_DIR,
            workspace_limits=self.workspace_limits,
            heightmap_resolution=self.heightmap_resolution,
        )

    def grasp(self, position: Sequence[float], heightmap_rotation_angle: float, workspace_limits) -> bool:
        print("Executing: grasp at (%f, %f, %f)" % (position[0], position[1], position[2]))
        before_positions = np.asarray(self.get_obj_positions(), dtype=np.float32)
        execute_grasp(
            target_xyz=position,
            theta=heightmap_rotation_angle,
            mp=self.mp,
            hand_link=self.hand_link,
            config=self.primitive_config,
            use_servo=False,
            capture_rgbd=lambda: capture_rgbd(self.camera, scene=self.controller.scene, intrinsics=self.cam_intrinsics),
        )
        self._step_scene(steps=40)
        after_positions = np.asarray(self.get_obj_positions(), dtype=np.float32)
        if after_positions.size == 0:
            return False

        lifted = after_positions[:, 2] > np.maximum(before_positions[:, 2] + 0.06, 0.08)
        grasp_success = bool(np.any(lifted))
        if grasp_success:
            grasped_idx = int(np.argmax(after_positions[:, 2]))
            self._move_object_to_discard(grasped_idx)
        return grasp_success

    def push(self, position: Sequence[float], heightmap_rotation_angle: float, workspace_limits) -> bool:
        print("Executing: push at (%f, %f, %f)" % (position[0], position[1], position[2]))
        execute_push(
            target_xyz=position,
            theta=heightmap_rotation_angle,
            mp=self.mp,
            hand_link=self.hand_link,
            config=self.primitive_config,
            use_servo=False,
            capture_rgbd=lambda: capture_rgbd(self.camera, scene=self.controller.scene, intrinsics=self.cam_intrinsics),
        )
        self._step_scene(steps=30)
        return True

    def restart_sim(self) -> None:
        self.mp.move_to_pose(OBSERVATION_POSE)
        self.add_objects()
        self._step_scene(steps=80)

    def add_objects(self) -> None:
        positions = self._sample_clutter_positions(self.num_obj)
        if not self.object_handles:
            for idx, position in enumerate(positions):
                cfg = copy.deepcopy(self.object_configs[idx % len(self.object_configs)])
                cfg["object_name"] = f"{cfg.get('object_name', 'block')}_{idx:02d}"
                cfg["position"] = [float(position[0]), float(position[1]), 0.02]
                self.object_handles.append(self.controller.add_object(cfg))
        else:
            for idx, obj in enumerate(self.object_handles):
                self._set_object_pose(obj, [float(positions[idx][0]), float(positions[idx][1]), 0.02])
        self._step_scene(steps=60)

    def check_sim(self) -> None:
        hand_p = np.asarray(self.hand_link.get_entity_pose().p, dtype=np.float32)
        margin = 0.3
        ok = (
            self.workspace_limits[0][0] - margin <= hand_p[0] <= self.workspace_limits[0][1] + margin
            and self.workspace_limits[1][0] - margin <= hand_p[1] <= self.workspace_limits[1][1] + margin
            and self.workspace_limits[2][0] <= hand_p[2] <= self.primitive_config.safe_z + margin
        )
        if not ok:
            print("Simulation unstable. Restarting SAPIEN environment.")
            self.restart_sim()

    def get_obj_positions(self):
        return [self._get_object_position(obj).tolist() for obj in self.object_handles]

    def _setup_robot(self):
        robot_config = copy.deepcopy(panda_config)
        robot_config["position"] = [0.0, 0.0, 0.0]
        robot = self.controller.add_robot(robot_config)
        safe_home_qpos = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785, 0.04, 0.04]
        robot.set_qpos(safe_home_qpos)
        for i, joint in enumerate(robot.active_joints):
            if i < len(safe_home_qpos):
                joint.set_drive_target(safe_home_qpos[i])
        self._step_robot(robot, steps=50)
        return robot

    def _update_camera_pose(self) -> None:
        camera_pose_world = get_camera_pose_world(self.hand_link.get_entity_pose(), DEFAULT_CAM_POSE_IN_HAND)
        self.cam_pose = vpg_camera_pose_from_sapien(camera_pose_world)

    def _sample_clutter_positions(self, count: int) -> list[tuple[float, float]]:
        xs = np.linspace(0.30, 0.50, 5)
        ys = np.linspace(-0.16, 0.18, 5)
        grid = [(float(x), float(y)) for x in xs for y in ys]
        self.random.shuffle(grid)
        positions = []
        for x, y in grid[:count]:
            positions.append((x + self.random.uniform(-0.01, 0.01), y + self.random.uniform(-0.01, 0.01)))
        return positions

    def _move_object_to_discard(self, obj_idx: int) -> None:
        discard = self.discard_origin + np.asarray([0.0, 0.05 * obj_idx, 0.0], dtype=np.float32)
        self._set_object_pose(self.object_handles[obj_idx], discard.tolist())
        self._step_scene(steps=20)

    def _get_object_position(self, obj) -> np.ndarray:
        if hasattr(obj, "get_pose"):
            return np.asarray(obj.get_pose().p, dtype=np.float32)
        if hasattr(obj, "get_root_pose"):
            return np.asarray(obj.get_root_pose().p, dtype=np.float32)
        if hasattr(obj, "get_entity_pose"):
            return np.asarray(obj.get_entity_pose().p, dtype=np.float32)
        raise AttributeError("Unsupported SAPIEN object pose API")

    def _set_object_pose(self, obj, position: Sequence[float]) -> None:
        import sapien.core as sapien

        pose = sapien.Pose(position, [1, 0, 0, 0])
        if hasattr(obj, "set_pose"):
            obj.set_pose(pose)
        elif hasattr(obj, "set_root_pose"):
            obj.set_root_pose(pose)
        else:
            raise AttributeError("Unsupported SAPIEN object pose API")

    def _step_scene(self, *, steps: int) -> None:
        self._step_robot(self.robot, steps=steps)

    def _step_robot(self, robot, *, steps: int) -> None:
        for i in range(steps):
            qf = robot.compute_passive_force(gravity=True, coriolis_and_centrifugal=True)
            robot.set_qf(qf)
            self.controller.scene.step()
            if i % 4 == 0 and self.controller.viewer:
                self.controller.scene.update_render()
                self.controller.viewer.render()
