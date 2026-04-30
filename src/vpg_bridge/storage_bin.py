from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class StorageBinConfig:
    center: tuple[float, float, float] = (0.30, -0.34, 0.0)
    inner_size: tuple[float, float] = (0.18, 0.14)
    wall_height: float = 0.09
    wall_thickness: float = 0.01
    bottom_thickness: float = 0.01
    drop_z: float = 0.17
    object_z: float = 0.04


DEFAULT_STORAGE_BIN = StorageBinConfig()


def storage_bin_drop_pose(config: StorageBinConfig = DEFAULT_STORAGE_BIN) -> tuple[float, float, float]:
    """End-effector pose used to release a grasped object into the bin."""
    x, y, _ = config.center
    return float(x), float(y), float(config.drop_z)


def storage_bin_object_pose(index: int, config: StorageBinConfig = DEFAULT_STORAGE_BIN) -> tuple[float, float, float]:
    """Compact slot inside the bin for training resets after successful grasps."""
    x0, y0, _ = config.center
    slot_x = min(0.045, config.inner_size[0] / 4.0)
    slot_y = min(0.035, config.inner_size[1] / 4.0)
    grid = [
        (-slot_x, -slot_y),
        (0.0, -slot_y),
        (slot_x, -slot_y),
        (-slot_x, 0.0),
        (0.0, 0.0),
        (slot_x, 0.0),
        (-slot_x, slot_y),
        (0.0, slot_y),
        (slot_x, slot_y),
    ]
    dx, dy = grid[index % len(grid)]
    layer = index // len(grid)
    return float(x0 + dx), float(y0 + dy), float(config.object_z + 0.045 * layer)


def add_storage_bin_to_scene(scene, config: StorageBinConfig = DEFAULT_STORAGE_BIN, name: str = "storage_bin"):
    import sapien

    builder = scene.create_actor_builder()
    wall_color = (0.08, 0.12, 0.16)
    bottom_color = (0.18, 0.22, 0.25)

    inner_x, inner_y = config.inner_size
    t = config.wall_thickness
    bottom_t = config.bottom_thickness
    wall_h = config.wall_height
    wall_z = bottom_t + wall_h / 2.0

    parts: list[tuple[Sequence[float], Sequence[float], Sequence[float]]] = [
        ((0.0, 0.0, bottom_t / 2.0), (inner_x / 2.0 + t, inner_y / 2.0 + t, bottom_t / 2.0), bottom_color),
        ((inner_x / 2.0 + t / 2.0, 0.0, wall_z), (t / 2.0, inner_y / 2.0 + t, wall_h / 2.0), wall_color),
        ((-inner_x / 2.0 - t / 2.0, 0.0, wall_z), (t / 2.0, inner_y / 2.0 + t, wall_h / 2.0), wall_color),
        ((0.0, inner_y / 2.0 + t / 2.0, wall_z), (inner_x / 2.0, t / 2.0, wall_h / 2.0), wall_color),
        ((0.0, -inner_y / 2.0 - t / 2.0, wall_z), (inner_x / 2.0, t / 2.0, wall_h / 2.0), wall_color),
    ]

    for local_pose, half_size, color in parts:
        pose = sapien.Pose(local_pose, [1, 0, 0, 0])
        builder.add_box_collision(pose=pose, half_size=half_size)
        builder.add_box_visual(pose=pose, half_size=half_size, material=color)

    builder.set_initial_pose(sapien.Pose(config.center, [1, 0, 0, 0]))
    return builder.build_static(name=name)
