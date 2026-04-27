from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .heightmap import DEFAULT_HEIGHTMAP_RESOLUTION, DEFAULT_WORKSPACE_LIMITS


@dataclass
class VPGAction:
    action: str
    target_xyz: np.ndarray
    theta: float
    confidence: float
    best_pix_ind: tuple
    push_confidence: float
    grasp_confidence: float


class VPGPolicy:
    def __init__(
        self,
        *,
        vpg_dir: Path,
        snapshot_file: Path,
        workspace_limits: np.ndarray = DEFAULT_WORKSPACE_LIMITS,
        heightmap_resolution: float = DEFAULT_HEIGHTMAP_RESOLUTION,
        force_cpu: bool = False,
    ) -> None:
        self.vpg_dir = Path(vpg_dir).resolve()
        self.snapshot_file = Path(snapshot_file).resolve()
        self.workspace_limits = np.asarray(workspace_limits, dtype=np.float32)
        self.heightmap_resolution = float(heightmap_resolution)

        if not self.snapshot_file.exists():
            raise FileNotFoundError(
                f"VPG snapshot not found: {self.snapshot_file}\n"
                "Download weights first, for example: "
                "cd visual-pushing-grasping-master/downloads && ./download-weights.sh"
            )

        if str(self.vpg_dir) not in sys.path:
            sys.path.insert(0, str(self.vpg_dir))

        self._patch_torchvision_densenet_for_offline_loading()
        try:
            trainer_module = importlib.import_module("trainer")
            torch = importlib.import_module("torch")
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "VPG inference requires PyTorch/torchvision in the active Python environment. "
                "Install the VPG dependencies before running src.vpg_visual_servo_main."
            ) from exc

        self.trainer = trainer_module.Trainer(
            method="reinforcement",
            push_rewards=True,
            future_reward_discount=0.5,
            is_testing=True,
            load_snapshot=False,
            snapshot_file=None,
            force_cpu=force_cpu,
        )

        map_location = "cpu" if force_cpu or not self.trainer.use_cuda else None
        state_dict = torch.load(str(self.snapshot_file), map_location=map_location)
        self.trainer.model.load_state_dict(state_dict)
        if self.trainer.use_cuda:
            self.trainer.model = self.trainer.model.cuda()
        self.trainer.model.eval()

    def predict(self, color_heightmap: np.ndarray, depth_heightmap: np.ndarray) -> VPGAction:
        push_predictions, grasp_predictions, _ = self.trainer.forward(
            color_heightmap,
            depth_heightmap,
            is_volatile=True,
        )

        push_confidence = float(np.max(push_predictions))
        grasp_confidence = float(np.max(grasp_predictions))
        if push_confidence > grasp_confidence:
            action = "push"
            best_pix_ind = np.unravel_index(np.argmax(push_predictions), push_predictions.shape)
            confidence = push_confidence
        else:
            action = "grasp"
            best_pix_ind = np.unravel_index(np.argmax(grasp_predictions), grasp_predictions.shape)
            confidence = grasp_confidence

        rotation_idx, pix_y, pix_x = [int(v) for v in best_pix_ind]
        theta = float(np.deg2rad(rotation_idx * (360.0 / self.trainer.model.num_rotations)))
        target_xyz = self.pixel_to_world(pix_x, pix_y, depth_heightmap[pix_y, pix_x])

        return VPGAction(
            action=action,
            target_xyz=target_xyz,
            theta=theta,
            confidence=confidence,
            best_pix_ind=(rotation_idx, pix_y, pix_x),
            push_confidence=push_confidence,
            grasp_confidence=grasp_confidence,
        )

    def pixel_to_world(self, pix_x: int, pix_y: int, depth_value: float) -> np.ndarray:
        x = pix_x * self.heightmap_resolution + self.workspace_limits[0][0]
        y = pix_y * self.heightmap_resolution + self.workspace_limits[1][0]
        z = float(depth_value) + self.workspace_limits[2][0]
        return np.asarray([x, y, z], dtype=np.float32)

    @staticmethod
    def _patch_torchvision_densenet_for_offline_loading() -> None:
        """Prevent torchvision from downloading ImageNet weights.

        The VPG snapshot provides the actual weights. The original model code
        asks torchvision for pretrained DenseNet trunks before loading the
        snapshot, which is inconvenient in offline lab setups.
        """
        try:
            densenet_module = importlib.import_module("torchvision.models.densenet")
        except Exception:
            return

        original = getattr(densenet_module, "densenet121", None)
        if original is None or getattr(original, "_vpg_offline_patch", False):
            return

        def densenet121_offline(*args, **kwargs):
            kwargs.pop("pretrained", None)
            kwargs["weights"] = None
            try:
                return original(*args, **kwargs)
            except TypeError:
                kwargs.pop("weights", None)
                kwargs["pretrained"] = False
                return original(*args, **kwargs)

        densenet121_offline._vpg_offline_patch = True
        densenet_module.densenet121 = densenet121_offline
