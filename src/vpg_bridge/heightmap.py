from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Tuple

import numpy as np


DEFAULT_WORKSPACE_LIMITS = np.asarray(
    [[0.176, 0.624], [-0.224, 0.224], [0.0, 0.4]],
    dtype=np.float32,
)
DEFAULT_HEIGHTMAP_RESOLUTION = 0.002


def get_vpg_utils(vpg_dir: Path):
    vpg_dir = Path(vpg_dir).resolve()
    if str(vpg_dir) not in sys.path:
        sys.path.insert(0, str(vpg_dir))
    return importlib.import_module("utils")


def build_heightmap(
    color_img: np.ndarray,
    depth_img: np.ndarray,
    intrinsics: np.ndarray,
    camera_pose: np.ndarray,
    *,
    vpg_dir: Path,
    workspace_limits: np.ndarray = DEFAULT_WORKSPACE_LIMITS,
    heightmap_resolution: float = DEFAULT_HEIGHTMAP_RESOLUTION,
) -> Tuple[np.ndarray, np.ndarray]:
    try:
        utils = get_vpg_utils(vpg_dir)
        color_heightmap, depth_heightmap = utils.get_heightmap(
            color_img,
            depth_img,
            intrinsics,
            camera_pose,
            workspace_limits,
            heightmap_resolution,
        )
    except ModuleNotFoundError:
        color_heightmap, depth_heightmap = _local_get_heightmap(
            color_img,
            depth_img,
            intrinsics,
            camera_pose,
            workspace_limits,
            heightmap_resolution,
        )
    valid_depth_heightmap = depth_heightmap.copy()
    valid_depth_heightmap[np.isnan(valid_depth_heightmap)] = 0.0
    return color_heightmap, valid_depth_heightmap


def _local_get_heightmap(
    color_img: np.ndarray,
    depth_img: np.ndarray,
    cam_intrinsics: np.ndarray,
    cam_pose: np.ndarray,
    workspace_limits: np.ndarray,
    heightmap_resolution: float,
) -> Tuple[np.ndarray, np.ndarray]:
    heightmap_size = np.round(
        (
            (workspace_limits[1][1] - workspace_limits[1][0]) / heightmap_resolution,
            (workspace_limits[0][1] - workspace_limits[0][0]) / heightmap_resolution,
        )
    ).astype(int)

    surface_pts, color_pts = _local_get_pointcloud(color_img, depth_img, cam_intrinsics)
    surface_pts = np.transpose(
        np.dot(cam_pose[0:3, 0:3], np.transpose(surface_pts))
        + np.tile(cam_pose[0:3, 3:], (1, surface_pts.shape[0]))
    )

    sort_z_ind = np.argsort(surface_pts[:, 2])
    surface_pts = surface_pts[sort_z_ind]
    color_pts = color_pts[sort_z_ind]

    valid_ind = np.logical_and.reduce(
        (
            surface_pts[:, 0] >= workspace_limits[0][0],
            surface_pts[:, 0] < workspace_limits[0][1],
            surface_pts[:, 1] >= workspace_limits[1][0],
            surface_pts[:, 1] < workspace_limits[1][1],
            surface_pts[:, 2] < workspace_limits[2][1],
        )
    )
    surface_pts = surface_pts[valid_ind]
    color_pts = color_pts[valid_ind]

    color_heightmap = np.zeros((heightmap_size[0], heightmap_size[1], 3), dtype=np.uint8)
    depth_heightmap = np.zeros(heightmap_size, dtype=np.float32)

    if surface_pts.shape[0] > 0:
        pix_x = np.floor((surface_pts[:, 0] - workspace_limits[0][0]) / heightmap_resolution).astype(int)
        pix_y = np.floor((surface_pts[:, 1] - workspace_limits[1][0]) / heightmap_resolution).astype(int)
        pix_valid = np.logical_and.reduce(
            (
                pix_x >= 0,
                pix_x < heightmap_size[1],
                pix_y >= 0,
                pix_y < heightmap_size[0],
            )
        )
        pix_x = pix_x[pix_valid]
        pix_y = pix_y[pix_valid]
        surface_pts = surface_pts[pix_valid]
        color_pts = color_pts[pix_valid]
        color_heightmap[pix_y, pix_x, :] = color_pts
        depth_heightmap[pix_y, pix_x] = surface_pts[:, 2]

    z_bottom = workspace_limits[2][0]
    depth_heightmap = depth_heightmap - z_bottom
    depth_heightmap[depth_heightmap < 0] = 0.0
    depth_heightmap[depth_heightmap == -z_bottom] = np.nan
    return color_heightmap, depth_heightmap


def _local_get_pointcloud(
    color_img: np.ndarray,
    depth_img: np.ndarray,
    camera_intrinsics: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    im_h, im_w = depth_img.shape[:2]
    pix_x, pix_y = np.meshgrid(np.linspace(0, im_w - 1, im_w), np.linspace(0, im_h - 1, im_h))
    cam_pts_x = np.multiply(pix_x - camera_intrinsics[0][2], depth_img / camera_intrinsics[0][0])
    cam_pts_y = np.multiply(pix_y - camera_intrinsics[1][2], depth_img / camera_intrinsics[1][1])
    cam_pts_z = depth_img.copy()
    cam_pts = np.stack((cam_pts_x, cam_pts_y, cam_pts_z), axis=2).reshape(-1, 3)
    rgb_pts = color_img[:, :, :3].reshape(-1, 3)
    return cam_pts, rgb_pts
