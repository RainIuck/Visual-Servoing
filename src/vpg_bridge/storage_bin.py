from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class StorageBinConfig:
    center: tuple[float, float, float] = (0.30, -0.34, 0.0)
    inner_size: tuple[float, float] = (0.18, 0.14)
    wall_height: float = 0.09
    wall_thickness: float = 0.01
    bottom_thickness: float = 0.01
    drop_z: float = 0.17


DEFAULT_STORAGE_BIN = StorageBinConfig()


def storage_bin_drop_pose(config: StorageBinConfig = DEFAULT_STORAGE_BIN) -> tuple[float, float, float]:
    x, y, _ = config.center
    return float(x), float(y), float(config.drop_z)


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
