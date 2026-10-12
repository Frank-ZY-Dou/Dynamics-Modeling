"""
GPU-accelerated data augmentation for training.

All augmentations operate on GPU tensors and are applied in the training
loop after the batch has been moved to the GPU.
"""

import torch
import math
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class GPUAugmentation:
    """
    GPU-accelerated augmentation pipeline.

    Features:
    - Rotation augmentation (Z-axis only, no ground mesh)
    - All operations on GPU tensors
    """

    def __init__(self, config: Dict, device: torch.device):
        """
        Initialize GPU augmentation.

        Args:
            config: Augmentation configuration
            device: Target GPU device
        """
        self.device = device
        self.config = config

        # Rotation augmentation settings
        rotation_cfg = config.get('data', {}).get('augmentation', {}).get('rotation_augmentation', {})
        self.rotation_enabled = rotation_cfg.get('enable', False)
        self.rotation_angle_step = rotation_cfg.get('angle_step_size', 30)
        self.rotation_prob = rotation_cfg.get('prob_use_rotation', 0.3)

        # Pre-compute rotation angles (only if rotation enabled and valid step size)
        if self.rotation_enabled and self.rotation_angle_step > 0:
            self.rotation_angles = torch.arange(
                self.rotation_angle_step, 360, self.rotation_angle_step,
                dtype=torch.float32, device=device
            )
        else:
            # Rotation disabled or invalid step size - create empty tensor
            self.rotation_angles = torch.tensor([], dtype=torch.float32, device=device)

        # Counters for periodic logging.
        self.rotation_total_calls = 0
        self.rotation_applied_count = 0

        logger.info(f"\n{'='*80}")
        logger.info(f"[GPU Augmentation] Initialization")
        logger.info(f"{'='*80}")
        logger.info(f"Rotation Augmentation:")
        logger.info(f"  - Enabled: {self.rotation_enabled}")
        if self.rotation_enabled:
            logger.info(f"  - Angle step size: {self.rotation_angle_step}°")
            logger.info(f"  - Rotation angles: {self.rotation_angles.cpu().tolist()}")
            logger.info(f"  - Probability: {self.rotation_prob:.2%}")
        logger.info(f"{'='*80}\n")

    def apply_rotation_augmentation(
        self,
        batch: Dict[str, torch.Tensor],
        split: str = 'train'
    ) -> Dict[str, torch.Tensor]:
        """
        Apply rotation augmentation around Z-axis (GPU version, no ground mesh).

        Args:
            batch: Batch dict with GPU tensors
            split: 'train', 'val', or 'test' (only train gets augmented)

        Returns:
            Augmented batch dict
        """
        # Only apply to training split
        if split != 'train':
            batch['rotation_angle_deg'] = 0.0
            return batch

        self.rotation_total_calls += 1

        if not self.rotation_enabled:
            batch['rotation_angle_deg'] = 0.0
            return batch

        if len(self.rotation_angles) == 0:
            raise ValueError(
                f"Rotation enabled but no valid angles available. "
                f"angle_step_size={self.rotation_angle_step} must be > 0"
            )

        # Decide whether to rotate (probabilistic)
        rand_val = torch.rand(1, device=self.device).item()
        if rand_val > self.rotation_prob:
            batch['rotation_angle_deg'] = 0.0
            return batch

        # Randomly select rotation angle
        angle_idx = torch.randint(0, len(self.rotation_angles), (1,), device=self.device)
        angle_deg = self.rotation_angles[angle_idx].item()

        if angle_deg == 0:
            batch['rotation_angle_deg'] = 0.0
            return batch

        self.rotation_applied_count += 1
        if self.rotation_applied_count <= 5 or self.rotation_total_calls % 100 == 0:
            apply_rate = self.rotation_applied_count / self.rotation_total_calls
            logger.info(
                f"[GPU Rotation Aug] Applying {angle_deg:.0f}° | "
                f"Batch {self.rotation_total_calls} | "
                f"Applied: {self.rotation_applied_count}/{self.rotation_total_calls} ({apply_rate:.2%})"
            )

        # Create rotation matrix (Z-axis)
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        R = torch.tensor([
            [cos_a, -sin_a, 0],
            [sin_a,  cos_a, 0],
            [0,      0,     1]
        ], dtype=torch.float32, device=self.device)

        # Rotate obj_positions (B, T, Nobj, Nv, 3); all objects rotate since
        # there is no ground mesh.
        obj_pos = batch['obj_positions']
        obj_pos_flat = obj_pos.reshape(-1, 3)
        obj_pos_rotated = torch.matmul(obj_pos_flat, R.T)
        batch['obj_positions'] = obj_pos_rotated.view(obj_pos.shape)

        # Rotate ref_obj_positions: (B, Nobj, Nv, 3)
        ref_pos = batch['ref_obj_positions']
        ref_pos_flat = ref_pos.reshape(-1, 3)
        ref_pos_rotated = torch.matmul(ref_pos_flat, R.T)
        batch['ref_obj_positions'] = ref_pos_rotated.view(ref_pos.shape)


        # Record the applied angle so ground-truth overlays can be rotated to match.
        batch['rotation_angle_deg'] = angle_deg

        return batch

    def __call__(
        self,
        batch: Dict[str, torch.Tensor],
        split: str = 'train'
    ) -> Dict[str, torch.Tensor]:
        """
        Apply all enabled augmentations.

        Args:
            batch: Batch dict with GPU tensors
            split: 'train', 'val', or 'test'

        Returns:
            Augmented batch dict
        """
        if self.rotation_enabled:
            batch = self.apply_rotation_augmentation(batch, split)
        else:
            # Rotation disabled: record a zero angle for consistency.
            batch['rotation_angle_deg'] = 0.0

        return batch
