"""
Random walk noise augmentation for physics simulation.
Based on Learning to Simulate (Sanchez-Gonzalez et al., 2020)
"""

import torch
import numpy as np
from typing import Optional, Tuple, Dict


def get_random_walk_noise_for_position_sequence(
    position_sequence: torch.Tensor,
    noise_std_last_step: float,
    dt: float = 1.0
) -> torch.Tensor:
    """
    Generate random walk noise for position sequences.

    This noise mimics the accumulation of errors during rollout,
    making the model more robust to its own prediction errors.

    Args:
        position_sequence: Tensor of shape (time, num_particles, 3) or (batch, time, num_particles, 3)
        noise_std_last_step: Standard deviation of noise at the last step
        dt: Time step for velocity computation

    Returns:
        position_noise: Tensor of same shape as position_sequence
    """
    # Handle both 3D and 4D tensors
    if position_sequence.dim() == 3:
        # Add batch dimension for consistency
        position_sequence = position_sequence.unsqueeze(0)
        squeeze_batch = True
    else:
        squeeze_batch = False

    batch_size, seq_len, num_particles, dim = position_sequence.shape

    # We want the noise scale in the velocity at the last step to be fixed.
    # Because we are going to compose noise at each step using a random walk:
    # std_last_step^2 = num_velocities * std_each_step^2
    # so to keep std_last_step fixed, we apply at each step:
    # std_each_step = std_last_step / sqrt(num_velocities)
    num_velocities = seq_len - 1  # Number of velocity steps

    if num_velocities <= 0:
        return torch.zeros_like(position_sequence)

    # Generate independent noise for each velocity step
    velocity_shape = (batch_size, num_velocities, num_particles, dim)
    std_each_step = noise_std_last_step / np.sqrt(num_velocities)

    # Generate velocity noise
    velocity_noise = torch.randn(
        velocity_shape,
        device=position_sequence.device,
        dtype=position_sequence.dtype
    ) * std_each_step

    # Apply random walk by cumulative sum
    velocity_noise_accumulated = torch.cumsum(velocity_noise, dim=1)

    # Integrate velocity noise to get position noise
    # First position has no noise, then cumsum of velocities
    position_noise = torch.zeros_like(position_sequence)
    if num_velocities > 0:
        # Cumulative sum of accumulated velocities gives position noise
        position_noise_from_vel = torch.cumsum(velocity_noise_accumulated * dt, dim=1)
        position_noise[:, 1:] = position_noise_from_vel

    # Remove batch dimension if we added it
    if squeeze_batch:
        position_noise = position_noise.squeeze(0)

    return position_noise


def apply_random_walk_noise_to_batch(
    batch: Dict[str, torch.Tensor],
    noise_config: Dict,
    is_training: bool = True
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Apply random walk noise to a batch of data.

    Args:
        batch: Dictionary containing 'body1_positions', 'body2_positions', etc.
        noise_config: Configuration dictionary with noise parameters
        is_training: Whether in training mode (no noise during validation)

    Returns:
        noisy_batch: Batch with noise applied to positions
        noise_info: Dictionary containing the applied noise for target correction
    """
    if not is_training or not noise_config.get('random_walk', {}).get('enable', False):
        return batch, {}

    noise_std = noise_config['random_walk'].get('noise_std', 0.0001)
    dt = noise_config.get('dt', 1.0)

    # Copy batch to avoid modifying original
    noisy_batch = {k: v.clone() if torch.is_tensor(v) else v for k, v in batch.items()}
    noise_info = {}

    # Apply noise to body1 and body2 positions
    for body_key in ['body1_positions', 'body2_positions']:
        if body_key in batch:
            positions = batch[body_key]

            # Generate random walk noise
            position_noise = get_random_walk_noise_for_position_sequence(
                positions, noise_std, dt
            )

            # Apply noise to positions
            noisy_batch[body_key] = positions + position_noise

            # Store noise for target correction
            noise_key = body_key.replace('positions', 'noise')
            noise_info[noise_key] = position_noise

            # Recompute velocities from noisy positions
            vel_key = body_key.replace('positions', 'velocities')
            if vel_key in batch:
                # Compute velocities from noisy positions
                noisy_positions = noisy_batch[body_key]

                # Handle both 3D (seq_len, num_particles, 3) and 4D tensors
                if noisy_positions.dim() == 3:
                    seq_len = noisy_positions.shape[0]
                    if seq_len > 1:
                        # Finite differences between consecutive noisy positions
                        velocities = torch.zeros_like(batch[vel_key])
                        vel_seq_len = velocities.shape[0]
                        if vel_seq_len > 0:
                            # First velocity from first two positions
                            velocities[0] = (noisy_positions[1] - noisy_positions[0]) / dt
                            # Middle velocities
                            for i in range(1, min(vel_seq_len-1, seq_len-2)):
                                velocities[i] = (noisy_positions[i+1] - noisy_positions[i]) / dt
                            # Last velocity if needed
                            if vel_seq_len > 1 and seq_len > 2:
                                velocities[-1] = (noisy_positions[-1] - noisy_positions[-2]) / dt
                        noisy_batch[vel_key] = velocities
                else:
                    # 4D case (batch dimension)
                    if noisy_positions.shape[1] > 1:
                        velocities = torch.zeros_like(batch[vel_key])
                        if velocities.shape[1] > 0:
                            velocities[:, 0] = (noisy_positions[:, 1] - noisy_positions[:, 0]) / dt
                            for i in range(1, min(velocities.shape[1]-1, noisy_positions.shape[1]-2)):
                                velocities[:, i] = (noisy_positions[:, i+1] - noisy_positions[:, i]) / dt
                            if velocities.shape[1] > 1 and noisy_positions.shape[1] > 2:
                                velocities[:, -1] = (noisy_positions[:, -1] - noisy_positions[:, -2]) / dt
                        noisy_batch[vel_key] = velocities

            # Recompute accelerations from noisy positions
            accel_key = body_key.replace('positions', 'accelerations')
            if accel_key in batch:
                noisy_positions = noisy_batch[body_key]

                # Handle both 3D and 4D tensors
                if noisy_positions.dim() == 3:
                    seq_len = noisy_positions.shape[0]
                    if seq_len >= 3:
                        # Compute accelerations using second-order differences
                        accelerations = torch.zeros_like(batch[accel_key])
                        accel_seq_len = accelerations.shape[0]
                        for i in range(accel_seq_len):
                            if i + 1 < seq_len - 1:
                                accelerations[i] = (
                                    noisy_positions[i+2] - 2 * noisy_positions[i+1] + noisy_positions[i]
                                ) / (dt * dt)
                        noisy_batch[accel_key] = accelerations
                else:
                    # 4D case (batch dimension)
                    if noisy_positions.shape[1] >= 3:
                        accelerations = torch.zeros_like(batch[accel_key])
                        for i in range(accelerations.shape[1]):
                            if i + 1 < noisy_positions.shape[1] - 1:
                                accelerations[:, i] = (
                                    noisy_positions[:, i+2] - 2 * noisy_positions[:, i+1] + noisy_positions[:, i]
                                ) / (dt * dt)
                        noisy_batch[accel_key] = accelerations

    return noisy_batch, noise_info


def adjust_targets_for_noise(
    targets: Dict[str, torch.Tensor],
    noise_info: Dict[str, torch.Tensor],
    gamma: float = 1.0
) -> Dict[str, torch.Tensor]:
    """
    Adjust targets to account for input noise, so the model learns to correct it.

    Args:
        targets: Dictionary containing target positions/accelerations
        noise_info: Dictionary containing the noise applied to inputs
        gamma: Correction factor (1.0 = full correction, 0.0 = no correction)

    Returns:
        adjusted_targets: Targets adjusted for noise correction
    """
    adjusted_targets = {k: v.clone() if torch.is_tensor(v) else v for k, v in targets.items()}

    # Adjust position targets
    for body in ['body1', 'body2']:
        noise_key = f'{body}_noise'
        pos_key = f'{body}_positions'

        if noise_key in noise_info and pos_key in targets:
            # The target should include the noise at the last input step
            # so the model learns to predict outputs that correct for the noise
            last_noise = noise_info[noise_key][:, -1] if noise_info[noise_key].shape[1] > 0 else 0
            adjusted_targets[pos_key] = targets[pos_key] + gamma * last_noise

    return adjusted_targets


def apply_independent_noise(
    batch: Dict[str, torch.Tensor],
    noise_config: Dict,
    is_training: bool = True
) -> Dict[str, torch.Tensor]:
    """
    Apply independent Gaussian noise to velocities and other features.
    Independent noise model.

    Args:
        batch: Input batch
        noise_config: Noise configuration
        is_training: Whether in training mode

    Returns:
        Noisy batch
    """
    if not is_training or not noise_config.get('independent', {}).get('enable', False):
        return batch

    noisy_batch = {k: v.clone() if torch.is_tensor(v) else v for k, v in batch.items()}

    # Apply velocity noise
    vel_noise_std = noise_config['independent'].get('velocity_noise', 0.0)
    if vel_noise_std > 0:
        for key in ['body1_velocities', 'body2_velocities']:
            if key in batch:
                noise = torch.randn_like(batch[key]) * vel_noise_std
                noisy_batch[key] = batch[key] + noise

    return noisy_batch