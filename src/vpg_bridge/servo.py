from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import sapien.core as sapien

from .camera import get_camera_pose_world, vpg_camera_pose_from_sapien


@dataclass
class ServoResult:
    delta_xy: np.ndarray
    target_pixel: np.ndarray | None
    desired_pixel: np.ndarray
    pixel_error: np.ndarray | None
    status: str

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def estimate_alignment_error(
    *,
    target_xyz: Sequence[float],
    current_ee_pose: sapien.Pose,
    cam_pose_in_hand: sapien.Pose,
    intrinsics: np.ndarray,
    depth_img: np.ndarray,
    desired_gripper_pixel: Optional[Sequence[float]] = None,
    max_xy_correction: float = 0.02,
) -> ServoResult:
    """Estimate a small execution correction using the wrist RGB-D projection.

    This does not search for a new object center. It projects the VPG target
    into the current wrist camera image and compares it with the calibrated
    gripper action pixel.
    """
    target_xyz = np.asarray(target_xyz, dtype=np.float32)
    depth_img = np.asarray(depth_img, dtype=np.float32)

    camera_pose_world = get_camera_pose_world(current_ee_pose, cam_pose_in_hand)
    cam_pose = vpg_camera_pose_from_sapien(camera_pose_world)
    world_to_cam = np.linalg.inv(cam_pose)
    target_cam_h = world_to_cam @ np.asarray([target_xyz[0], target_xyz[1], target_xyz[2], 1.0], dtype=np.float32)
    target_cam = target_cam_h[:3]
    if desired_gripper_pixel is None:
        desired_world = np.asarray(
            [current_ee_pose.p[0], current_ee_pose.p[1], target_xyz[2], 1.0],
            dtype=np.float32,
        )
        desired_cam = (world_to_cam @ desired_world)[:3]
        if not np.all(np.isfinite(desired_cam)) or desired_cam[2] <= 1e-6:
            return _zero_result(np.asarray([0.0, 0.0], dtype=np.float32), "desired_point_behind_camera")
        desired_pixel = project_camera_point(desired_cam, intrinsics)
    else:
        desired_pixel = np.asarray(desired_gripper_pixel, dtype=np.float32)

    if not np.all(np.isfinite(target_cam)) or target_cam[2] <= 1e-6:
        return _zero_result(desired_pixel, "target_behind_camera")

    target_pixel = project_camera_point(target_cam, intrinsics)
    height, width = depth_img.shape[:2]
    if not _pixel_in_bounds(target_pixel, width, height):
        return _zero_result(desired_pixel, "target_projection_out_of_bounds", target_pixel=target_pixel)

    observed_depth = sample_depth(depth_img, target_pixel)
    if not np.isfinite(observed_depth) or observed_depth <= 0.0:
        return _zero_result(desired_pixel, "invalid_depth_at_target_pixel", target_pixel=target_pixel)

    pixel_error = target_pixel - desired_pixel
    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    if abs(fx) <= 1e-6 or abs(fy) <= 1e-6:
        return _zero_result(desired_pixel, "invalid_intrinsics", target_pixel=target_pixel)

    delta_cam = np.asarray(
        [
            pixel_error[0] * observed_depth / fx,
            pixel_error[1] * observed_depth / fy,
            0.0,
        ],
        dtype=np.float32,
    )
    delta_world = cam_pose[:3, :3] @ delta_cam
    delta_xy = delta_world[:2].astype(np.float32)

    if not np.all(np.isfinite(delta_xy)):
        return _zero_result(desired_pixel, "nonfinite_correction", target_pixel=target_pixel, pixel_error=pixel_error)

    if float(np.linalg.norm(delta_xy)) > max_xy_correction:
        return _zero_result(desired_pixel, "correction_exceeds_limit", target_pixel=target_pixel, pixel_error=pixel_error)

    return ServoResult(
        delta_xy=delta_xy,
        target_pixel=target_pixel,
        desired_pixel=desired_pixel,
        pixel_error=pixel_error.astype(np.float32),
        status="ok",
    )


def project_camera_point(point_camera: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    x, y, z = [float(v) for v in point_camera]
    u = float(intrinsics[0, 0]) * x / z + float(intrinsics[0, 2])
    v = float(intrinsics[1, 1]) * y / z + float(intrinsics[1, 2])
    return np.asarray([u, v], dtype=np.float32)


def sample_depth(depth_img: np.ndarray, pixel: np.ndarray) -> float:
    u = int(round(float(pixel[0])))
    v = int(round(float(pixel[1])))
    height, width = depth_img.shape[:2]
    if u < 0 or u >= width or v < 0 or v >= height:
        return 0.0
    return float(depth_img[v, u])


def _pixel_in_bounds(pixel: np.ndarray, width: int, height: int) -> bool:
    return 0.0 <= float(pixel[0]) < width and 0.0 <= float(pixel[1]) < height


def _zero_result(
    desired_pixel: np.ndarray,
    status: str,
    *,
    target_pixel: np.ndarray | None = None,
    pixel_error: np.ndarray | None = None,
) -> ServoResult:
    return ServoResult(
        delta_xy=np.zeros(2, dtype=np.float32),
        target_pixel=target_pixel,
        desired_pixel=desired_pixel,
        pixel_error=pixel_error,
        status=status,
    )
