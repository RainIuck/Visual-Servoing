from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import sapien.core as sapien


DEFAULT_CAM_POSE_IN_HAND = sapien.Pose([0.05, 0.0, 0.04], [0.7071, 0, -0.7071, 0])


@dataclass
class RGBDFrame:
    color: np.ndarray
    depth: np.ndarray
    intrinsics: np.ndarray


def find_link(robot, link_name: str):
    matches = [link for link in robot.get_links() if link.name == link_name]
    if not matches:
        raise ValueError(f"Robot link not found: {link_name}")
    return matches[0]


def _get_link_entity(link):
    """Return the SAPIEN entity that owns an articulation link component."""
    entity = getattr(link, "entity", None)
    return entity if entity is not None else link


def create_wrist_rgbd_camera(
    scene,
    robot,
    *,
    link_name: str = "panda_hand",
    pose_in_hand: sapien.Pose = DEFAULT_CAM_POSE_IN_HAND,
    width: int = 640,
    height: int = 480,
    fovy: float = 1.0,
    near: float = 0.01,
    far: float = 10.0,
):
    """Create a mounted RGB-D camera on the gripper hand link."""
    hand_link = find_link(robot, link_name)
    hand_entity = _get_link_entity(hand_link)
    if hasattr(scene, "add_mounted_camera"):
        camera = scene.add_mounted_camera(
            name="vpg_wrist_rgbd",
            mount=hand_entity,
            pose=pose_in_hand,
            width=width,
            height=height,
            fovy=fovy,
            near=near,
            far=far,
        )
    else:
        import sapien.render

        camera = sapien.render.RenderCameraComponent(width, height)
        if hasattr(camera, "set_fovy"):
            camera.set_fovy(fovy, compute_x=True)
        camera.near = near
        camera.far = far
        camera.local_pose = pose_in_hand
        hand_entity.add_component(camera)

    intrinsics = get_camera_intrinsics(camera, width=width, height=height, fovy=fovy)
    return camera, hand_link, intrinsics


def get_camera_intrinsics(camera, *, width: int, height: int, fovy: float) -> np.ndarray:
    if hasattr(camera, "get_intrinsic_matrix"):
        return np.asarray(camera.get_intrinsic_matrix(), dtype=np.float32)
    fy = height / (2.0 * np.tan(fovy / 2.0))
    fx = fy
    cx = width / 2.0
    cy = height / 2.0
    return np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)


def sapien_pose_to_matrix(pose: sapien.Pose) -> np.ndarray:
    if hasattr(pose, "to_transformation_matrix"):
        return np.asarray(pose.to_transformation_matrix(), dtype=np.float32)
    mat = np.eye(4, dtype=np.float32)
    mat[:3, :3] = pose.to_transformation_matrix()[:3, :3]
    mat[:3, 3] = pose.p
    return mat


def vpg_camera_pose_from_sapien(camera_pose_world: sapien.Pose) -> np.ndarray:
    """Return a camera-to-world matrix in VPG pinhole coordinates.

    VPG uses x=right, y=down, z=forward. The SAPIEN camera frame used by the
    mounted pose is x=forward, y=left, z=up, so this rotates VPG camera points
    into the SAPIEN camera frame before applying the world transform.
    """
    sapien_cam_to_world = sapien_pose_to_matrix(camera_pose_world)
    vpg_to_sapien = np.asarray(
        [
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=np.float32,
    )
    cam_pose = np.eye(4, dtype=np.float32)
    cam_pose[:3, :3] = sapien_cam_to_world[:3, :3] @ vpg_to_sapien
    cam_pose[:3, 3] = sapien_cam_to_world[:3, 3]
    return cam_pose


def get_camera_pose_world(hand_pose_world: sapien.Pose, pose_in_hand: sapien.Pose) -> sapien.Pose:
    return hand_pose_world * pose_in_hand


def capture_rgbd(camera, scene=None, intrinsics: Optional[np.ndarray] = None) -> RGBDFrame:
    if scene is not None:
        scene.update_render()
    camera.take_picture()
    color = _get_color(camera)
    depth = _get_depth(camera)
    if intrinsics is None:
        intrinsics = _intrinsics_from_camera_or_image(camera, color)
    return RGBDFrame(color=color, depth=depth, intrinsics=intrinsics)


def _get_color(camera) -> np.ndarray:
    rgba = camera.get_picture("Color") if hasattr(camera, "get_picture") else camera.get_color_rgba()
    rgba = np.asarray(rgba)
    color = (rgba[:, :, :3] * 255.0).clip(0, 255).astype(np.uint8)
    return color


def _get_depth(camera) -> np.ndarray:
    for name in ("Position", "Depth"):
        try:
            picture = np.asarray(camera.get_picture(name))
        except Exception:
            continue
        depth = _depth_from_picture(picture)
        if depth is not None:
            return depth
    raise RuntimeError("Unable to read depth from camera; expected a Position or Depth picture.")


def _depth_from_picture(picture: np.ndarray) -> Optional[np.ndarray]:
    if picture.ndim == 2:
        depth = picture.astype(np.float32)
        depth[~np.isfinite(depth)] = 0.0
        return depth

    if picture.ndim != 3:
        return None

    channels = picture[:, :, :3].astype(np.float32)
    candidates: Tuple[np.ndarray, ...] = (
        channels[:, :, 0],
        -channels[:, :, 2],
        channels[:, :, 2],
    )
    best = None
    best_count = 0
    for candidate in candidates:
        depth = candidate.copy()
        depth[~np.isfinite(depth)] = 0.0
        valid = np.logical_and(depth > 0.0, depth < 10.0)
        count = int(np.count_nonzero(valid))
        if count > best_count:
            best = depth
            best_count = count

    if best is None or best_count == 0:
        return None
    best[best < 0.0] = 0.0
    return best.astype(np.float32)


def _intrinsics_from_camera_or_image(camera, color: np.ndarray) -> np.ndarray:
    height, width = color.shape[:2]
    if hasattr(camera, "get_intrinsic_matrix"):
        return np.asarray(camera.get_intrinsic_matrix(), dtype=np.float32)
    fy = height / (2.0 * np.tan(1.0 / 2.0))
    return np.asarray(
        [[fy, 0.0, width / 2.0], [0.0, fy, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
