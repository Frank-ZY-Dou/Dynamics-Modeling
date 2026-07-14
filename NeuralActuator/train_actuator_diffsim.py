import argparse
import sys
import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx
import numpy as np
import pandas as pd
import optax
import os
import time
import json
import yaml
import flax.linen as nn
from flax.training import train_state
from torch.utils.tensorboard import SummaryWriter
import pickle
from tqdm import tqdm

from models import create_model, get_model_type_from_config
from evaluate_actuator import evaluate_on_csv, evaluate_batch_mjx, load_csv_data, build_features

# Feature normalization stats (set in main() when normalize_features is enabled).
# Saved inside every checkpoint so evaluation/inference reuse the same statistics.
_NORM_STATS = None
# EMA of parameters (set in main() when ema_decay > 0); saved alongside raw params.
_EMA_PARAMS = None


def _checkpoint_payload(params):
    if _NORM_STATS is None and _EMA_PARAMS is None:
        return params
    payload = {'params': jax.device_get(params)}
    if _NORM_STATS is not None:
        payload['feature_mean'] = np.asarray(_NORM_STATS[0])
        payload['feature_std'] = np.asarray(_NORM_STATS[1])
    if _EMA_PARAMS is not None:
        payload['ema_params'] = jax.device_get(_EMA_PARAMS)
    return payload


class TrainState(train_state.TrainState):
    pass

def load_dataset(csv_paths, mj_model, downsample_factor=1, return_boundaries=False, cfg=None):
    """Load dataset from CSV files with optional downsampling.

    Args:
        csv_paths: List of CSV file paths
        mj_model: MuJoCo model
        downsample_factor: Take every N-th row (1 = no downsampling)
        return_boundaries: If True, also return trajectory boundary indices
        cfg: Config dict (optional, for force_sentinel_interpolate option)

    Returns:
        If return_boundaries=False: (data_values, q_traj, v_traj, gt_pos, gt_force, force_valid)
        If return_boundaries=True: (data_values, q_traj, v_traj, gt_pos, gt_force, force_valid, boundaries)
            boundaries: List of (start_idx, end_idx) tuples for each trajectory

    36D Feature Vector (with position error and goal_aperture)
    """
    # 36D Feature Vector:
    # 0-4:   goal_pos1-5 (target joint positions, CONTROL SIGNAL from CSV)
    # 5-8:   pos1-4 (current joint positions, from simulation)
    # 9:     aperture (gripper aperture, from simulation)
    # 10-14: current1-5 (motor currents, from CSV, in mA)
    # 15-19: vel1-5 (joint velocities, from simulation)
    # 20-24: volts1-5 (motor voltages, from CSV)
    # 25-29: temp1-5 (motor temperatures, from CSV)
    # 30:    goal_aperture (target gripper aperture in mm, from CSV)
    # 31-34: error1-4 (goal_pos[:4] - pos[:4], position error for arm joints)
    # 35:    gripper_error (goal_aperture - aperture, gripper position error in mm)
    feature_cols = [
        'goal_pos1', 'goal_pos2', 'goal_pos3', 'goal_pos4', 'goal_pos5',  # Control signal
        'pos1', 'pos2', 'pos3', 'pos4', 'aperture',
        'current1', 'current2', 'current3', 'current4', 'current5',
        'vel1', 'vel2', 'vel3', 'vel4', 'vel5',
        'volts1', 'volts2', 'volts3', 'volts4', 'volts5',
        'temp1', 'temp2', 'temp3', 'temp4', 'temp5',
        'goal_aperture'  # Target gripper aperture (mm)
    ]

    data_values_all = []
    q_traj_all = []
    v_traj_all = []
    gt_pos_all = []
    gt_force_all = []
    force_valid_all = []  # Per-channel mask (n, 3): 1 where channel is valid, 0 where -999 (no sensor)

    for csv_path in csv_paths:
        try:
            print(f"Loading {csv_path}...")
            df = pd.read_csv(csv_path)
            df.columns = [c.strip() for c in df.columns]

            # Apply downsampling
            if downsample_factor > 1:
                orig_len = len(df)
                df = df.iloc[::downsample_factor].reset_index(drop=True)
                print(f"  Downsampled: {orig_len} -> {len(df)} rows (factor={downsample_factor})")
            
            # Check for missing goal columns - use fallback only if config allows
            goal_cols = ['goal_pos1', 'goal_pos2', 'goal_pos3', 'goal_pos4', 'goal_pos5', 'goal_aperture']
            missing_goal_cols = [c for c in goal_cols if c not in df.columns]
            if missing_goal_cols:
                if cfg is not None and cfg.get('use_goal_fallback', False):
                    print(f"  Warning: Missing goal columns {missing_goal_cols}. Using current positions as goals (use_goal_fallback=true).")
                    # Create goal columns from current positions
                    if 'goal_pos1' not in df.columns:
                        df['goal_pos1'] = df['pos1']
                    if 'goal_pos2' not in df.columns:
                        df['goal_pos2'] = df['pos2']
                    if 'goal_pos3' not in df.columns:
                        df['goal_pos3'] = df['pos3']
                    if 'goal_pos4' not in df.columns:
                        df['goal_pos4'] = df['pos4']
                    if 'goal_pos5' not in df.columns:
                        df['goal_pos5'] = df.get('pos5', df['aperture'])
                    if 'goal_aperture' not in df.columns:
                        df['goal_aperture'] = df['aperture']
                else:
                    raise KeyError(f"Missing required goal columns {missing_goal_cols}. Set 'use_goal_fallback: true' in config to use current positions as goals.")
            
            data_values = df[feature_cols].values
            # NOTE: current values stay in mA (indices 10-14)
            # Conversion to A happens in residual torque calculation: base_torque = (current_mA / 1000) * torque_constant

            # Compute error features:
            # arm error: goal_pos[:4] - pos[:4] (indices 0-3 minus 5-8)
            # gripper_error: goal_aperture - aperture (index 30 minus 9)
            arm_error = data_values[:, 0:4] - data_values[:, 5:9]  # shape: (n, 4)
            gripper_error = data_values[:, 30:31] - data_values[:, 9:10]  # shape: (n, 1), in mm
            data_values = np.concatenate([data_values, arm_error, gripper_error], axis=1)  # 31D -> 36D

            # Optional causal low-pass on the current channels (per trajectory,
            # same filter must be applied at evaluation/deployment time)
            lp_alpha = float(cfg.get('current_lowpass_alpha', 0.0)) if cfg else 0.0
            if lp_alpha > 0:
                cur = data_values[:, 10:15].copy()
                for t in range(1, len(cur)):
                    cur[t] = lp_alpha * cur[t] + (1.0 - lp_alpha) * cur[t - 1]
                data_values[:, 10:15] = cur

            n_samples = len(data_values)
            
            gt_pos = df[['pos1', 'pos2', 'pos3', 'pos4']].values
            apertures = df['aperture'].values
            # Convert aperture (mm) to gripper slide joint position (m)
            # MuJoCo gripper is a SLIDE joint with range [-0.011, 0.02] meters = [-11, 20] mm
            # CSV aperture = single finger position (mm), directly corresponds to MuJoCo joint
            gripper_q = apertures / 1000.0
            print(f"  Gripper GT: aperture {apertures.min():.1f}-{apertures.max():.1f}mm -> q {gripper_q.min():.4f}-{gripper_q.max():.4f}m")
            # Concatenate to make 5D GT
            gt_pos = np.column_stack([gt_pos, gripper_q])

            # Load GT Force (force_x, force_y, force_z)
            force_cols = ['force_x', 'force_y', 'force_z']
            if not all(col in df.columns for col in force_cols):
                print(f"Warning: Missing force columns in {csv_path}. Filling with zeros, force_valid=0.")
                gt_force = np.zeros((n_samples, 3))
                force_valid = np.zeros((n_samples, 3))  # All invalid (no force sensor)
            else:
                # Read force data, convert N/A to NaN, then to numeric
                force_df = df[force_cols].replace('N/A', np.nan).apply(pd.to_numeric, errors='coerce')

                # Create per-channel force_valid mask (n, 3) BEFORE converting -999 to 0
                # force_valid[i, c] = 1 where channel c is NOT -999 (some setups only
                # have a real force_z while x/y stay -999 throughout)
                sentinel_mask = (force_df == -999).values  # (n, 3), True where channel is -999
                sentinel_count = int(sentinel_mask.any(axis=1).sum())

                # Check if we should interpolate -999 sentinel values (for sparse force readings)
                if cfg is not None and cfg.get('force_sentinel_interpolate', False) and sentinel_count > 0:
                    # Convert -999 to NaN for interpolation
                    print(f"  Found {sentinel_count} samples with -999 sentinel -> will interpolate")
                    force_df = force_df.replace(-999, np.nan)
                    # Interpolate NaN values (linear interpolation between valid values)
                    force_df = force_df.interpolate(method='linear')
                    # Fill any remaining NaN (e.g., edges) with forward/backward fill, then 0
                    force_df = force_df.ffill().bfill().fillna(0.0)
                    # Assert no -999 values remain after interpolation
                    assert (force_df == -999).sum().sum() == 0, "ERROR: -999 values remain after interpolation!"
                    # After interpolation, all values are valid
                    force_valid = np.ones((n_samples, 3), dtype=np.float32)
                    print(f"  Interpolated {sentinel_count} sentinel values -> force_valid=1 for all")
                else:
                    force_valid = (~sentinel_mask).astype(np.float32)  # 1 = valid, 0 = invalid
                    if sentinel_count > 0:
                        print(f"  Found {sentinel_count} samples with -999 sentinel (no force sensor) -> force_valid=0")
                        # Convert -999 to 0 for safe math (but loss will be masked)
                        force_df = force_df.replace(-999, 0.0)

                nan_count = force_df.isna().sum().sum()

                if nan_count > 0:
                    print(f"  Found {nan_count} NaN values in force data.")
                    if cfg is not None and cfg.get('force_sentinel_interpolate', False):
                        # Interpolate NaN values (linear interpolation between valid values only)
                        force_df_interp = force_df.interpolate(method='linear')
                        # Fill any remaining NaN (e.g., edges) with 0
                        force_df_interp = force_df_interp.fillna(0.0)
                        gt_force = force_df_interp.values
                        print(f"  Interpolated {nan_count} NaN values")
                    else:
                        # Fill NaN with 0 (no interpolation)
                        gt_force = force_df.fillna(0.0).values
                        print(f"  NaN values filled with 0 (no interpolation)")
                else:
                    gt_force = force_df.values
            
            q_traj = np.zeros((n_samples, mj_model.nq))
            v_traj = np.zeros((n_samples, mj_model.nv))
            
            q_traj[:, :5] = gt_pos
            # Right gripper (equality) - set same as left (index 4)
            q_traj[:, 5] = gt_pos[:, 4]
            
            timestamps = df['timestamp'].values
            dt_data = np.mean(np.diff(timestamps))
            v_traj[:-1, :5] = (gt_pos[1:] - gt_pos[:-1]) / dt_data
            v_traj[:-1, 5] = v_traj[:-1, 4] # Copy velocity for right gripper
            
            data_values_all.append(data_values)
            q_traj_all.append(q_traj)
            v_traj_all.append(v_traj)
            gt_pos_all.append(gt_pos)
            gt_force_all.append(gt_force)
            force_valid_all.append(force_valid)
            
        except KeyError as e:
            if 'goal_pos' in str(e) or 'goal_aperture' in str(e):
                # This shouldn't happen if use_goal_fallback is properly set
                print(f"Error loading {csv_path}: {e}")
                print(f"  -> Set 'use_goal_fallback: true' in config to use current positions as goals.")
            else:
                print(f"Error loading {csv_path}: {e}")
        except Exception as e:
            print(f"Error loading {csv_path}: {e}")
            
    if not data_values_all:
        if return_boundaries:
            return None, None, None, None, None, None, None
        return None, None, None, None, None, None

    # Compute trajectory boundaries BEFORE concatenation
    boundaries = []
    current_idx = 0
    for data in data_values_all:
        traj_len = len(data)
        boundaries.append((current_idx, current_idx + traj_len))
        current_idx += traj_len

    print(f"  Loaded {len(boundaries)} trajectories with boundaries: {boundaries}")

    # Concatenate all data
    data_values = np.concatenate(data_values_all, axis=0)
    q_traj = np.concatenate(q_traj_all, axis=0)
    v_traj = np.concatenate(v_traj_all, axis=0)
    gt_pos = np.concatenate(gt_pos_all, axis=0)
    gt_force = np.concatenate(gt_force_all, axis=0)
    force_valid = np.concatenate(force_valid_all, axis=0)

    # Print force_valid summary (per channel)
    valid_ratio = force_valid.mean(axis=0)
    print(f"  Force valid ratio per channel: x={valid_ratio[0]:.1%} y={valid_ratio[1]:.1%} "
          f"z={valid_ratio[2]:.1%} ({len(force_valid)} samples)")

    if return_boundaries:
        return data_values, q_traj, v_traj, gt_pos, gt_force, force_valid, boundaries
    return data_values, q_traj, v_traj, gt_pos, gt_force, force_valid


def sample_valid_indices(boundaries, history_length, rollout_steps, batch_size, rng=None, debug=False, uniform_traj=True):
    """Sample start indices that respect trajectory boundaries, with zero-padding support.

    Sampling strategy:
    1. First uniformly random select a trajectory (file)
    2. Then sample start index within that trajectory's valid range
    3. History buffer uses zero-padding for samples near trajectory start

    Ensures that for each sampled start_idx:
    - rollout window [start_idx, start_idx + rollout_steps + 1) is within same trajectory
    - history buffer may use zero-padding if start_idx < traj_start + history_length

    Args:
        boundaries: List of (start_idx, end_idx) tuples for each trajectory
        history_length: Number of history frames needed
        rollout_steps: Number of rollout steps
        batch_size: Number of indices to sample
        rng: Optional numpy random generator
        debug: If True, print detailed debug info
        uniform_traj: If True, uniformly select trajectory first (default).
                      If False, weight by valid range size.

    Returns:
        (start_indices, traj_starts):
            start_indices: np.array of start indices
            traj_starts: np.array of trajectory start indices (for zero-padding calculation)
    """
    if rng is None:
        rng = np.random.default_rng()

    if debug:
        print(f"\n[DEBUG sample_valid_indices] (with zero-padding support)")
        print(f"  history_length={history_length}, rollout_steps={rollout_steps}, batch_size={batch_size}")
        print(f"  boundaries ({len(boundaries)} trajectories): {boundaries}")
        print(f"  uniform_traj={uniform_traj}")

    # Compute valid range for each trajectory
    # With zero-padding, start_idx can be anywhere in trajectory as long as rollout fits
    # min_start = traj_start (NOT traj_start + history_length anymore!)
    # max_start = traj_end - rollout_steps - 1
    valid_ranges = []
    for i, (traj_start, traj_end) in enumerate(boundaries):
        traj_len = traj_end - traj_start
        min_start = traj_start  # Can start from beginning (will use zero-padding for history)
        max_start = traj_end - rollout_steps - 1  # -1 for target offset
        valid_size = max_start - min_start

        if debug:
            print(f"  Traj {i}: [{traj_start}, {traj_end}) len={traj_len}, "
                  f"valid_range=[{min_start}, {max_start}) size={valid_size}")

        if max_start > min_start:
            valid_ranges.append((i, traj_start, min_start, max_start))  # Store traj_start too
        elif debug:
            print(f"    -> SKIPPED (too short, need {rollout_steps + 2} samples)")

    if not valid_ranges:
        raise ValueError(
            f"No valid sampling ranges! Each trajectory must have at least "
            f"{rollout_steps + 2} samples. "
            f"Boundaries: {boundaries}"
        )

    # Compute weights
    range_sizes = np.array([max_s - min_s for _, _, min_s, max_s in valid_ranges])
    total_valid = range_sizes.sum()

    if uniform_traj:
        # Uniform trajectory selection: each trajectory has equal probability
        weights = np.ones(len(valid_ranges)) / len(valid_ranges)
    else:
        # Weight by valid range size (longer trajectories sampled more)
        weights = range_sizes / total_valid

    if debug:
        print(f"  Valid ranges (traj_idx, traj_start, min, max): {valid_ranges}")
        print(f"  Range sizes: {range_sizes}, total={total_valid}")
        print(f"  Sampling weights: {weights}")

    # Sample trajectory indices according to weights
    valid_range_indices = rng.choice(len(valid_ranges), size=batch_size, p=weights)

    # Sample start index within each selected trajectory's valid range
    start_indices = np.zeros(batch_size, dtype=np.int32)
    traj_starts = np.zeros(batch_size, dtype=np.int32)  # Track trajectory start for zero-padding

    for i, vr_idx in enumerate(valid_range_indices):
        traj_idx, traj_start, min_start, max_start = valid_ranges[vr_idx]
        traj_starts[i] = traj_start
        start_indices[i] = rng.integers(min_start, max_start)

    if debug:
        print(f"  Sampled traj_starts: {traj_starts}")
        print(f"  Sampled start_indices: {start_indices}")
        # Verify each sample
        for i, (start_idx, traj_start) in enumerate(zip(start_indices, traj_starts)):
            history_start = start_idx - history_length
            rollout_end = start_idx + rollout_steps + 1
            # Find which trajectory this belongs to
            for t_idx, (t_start, t_end) in enumerate(boundaries):
                if t_start <= start_idx < t_end:
                    needs_padding = history_start < t_start
                    pad_len = max(0, t_start - history_start)
                    rollout_ok = rollout_end <= t_end
                    status = "OK" if rollout_ok else "ERROR!"
                    padding_info = f"(padding {pad_len} zeros)" if needs_padding else "(no padding)"
                    print(f"    Sample {i}: start={start_idx}, history=[{history_start},{start_idx}) {padding_info}, "
                          f"rollout=[{start_idx},{rollout_end}) in traj {t_idx} [{t_start},{t_end}) -> {status}")
                    break

    return start_indices, traj_starts


def validate_mujoco_joint_limits(mj_model, data_values, margin=0.05):
    """Validate that data joint positions fit within MuJoCo joint limits.

    Positions outside the model limits trigger explosive constraint forces in
    simulation, so training refuses to start on any mismatch (NO FALLBACKS).

    Args:
        mj_model: MuJoCo model (4 arm hinge joints + gripper slide joints)
        data_values: (N, 36) feature array; pos1-4 at indices 5-8 (rad)
        margin: Required margin between data range and model limits (rad)
    """
    print("=" * 60)
    print("[VALIDATION] MuJoCo Joint Limits vs Data Ranges")
    print("=" * 60)
    failures = []
    # Arm hinge joints only; the gripper slide joint is clamped explicitly
    # inside the rollout, so out-of-range apertures cannot diverge.
    for j in range(4):
        joint_name = mj_model.joint(j).name
        lo, hi = mj_model.jnt_range[j]
        data_lo = float(data_values[:, 5 + j].min())
        data_hi = float(data_values[:, 5 + j].max())
        ok = (data_lo >= lo + margin) and (data_hi <= hi - margin)
        print(f"  {joint_name}:")
        print(f"    Model limits: [{np.rad2deg(lo):.1f}°, {np.rad2deg(hi):.1f}°]")
        print(f"    Data range:   [{np.rad2deg(data_lo):.1f}°, {np.rad2deg(data_hi):.1f}°]")
        print(f"    Status: {'OK' if ok else 'RANGE MISMATCH'}")
        if not ok:
            failures.append(joint_name)
    if failures:
        raise ValueError(
            f"Joint limit validation failed for {failures}: data range within "
            f"{margin} rad of (or beyond) model limits. Update joint ranges in the robot MJCF.")
    print("  All joint limits validated successfully!")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Train Neural Actuator via Diff Sim')
    parser.add_argument('--csv', type=str, default=None, help='Path to trajectory CSV')
    parser.add_argument('--train_config', type=str, default='configs/weight_all.yaml', help='Path to training config YAML')
    parser.add_argument('--epochs', type=int, default=None, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--rollout_steps', type=int, default=None, help='Steps per rollout')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--log_dir', type=str, default=None, help='TensorBoard log dir')
    parser.add_argument('--model_out', type=str, default='outputs/neural_actuator_params.pkl', help='Path to save model')
    parser.add_argument('--pretrained_path', type=str, default=None, help='Path to pretrained params (optional)')
    parser.add_argument('--seed', type=int, default=0, help='Random seed (the seed key in the config takes precedence)')
    args = parser.parse_args()

    # 1. Load Config
    print(f"Loading config from {args.train_config}...")
    with open(args.train_config, 'r') as f:
        train_config = yaml.safe_load(f)
        
    epochs = args.epochs if args.epochs is not None else train_config['epochs']
    batch_size = args.batch_size if args.batch_size is not None else train_config['batch_size']
    rollout_steps = args.rollout_steps if args.rollout_steps is not None else train_config['rollout_steps']
    lr = args.lr if args.lr is not None else float(train_config['lr'])
    # Append timestamp to log_dir
    # Create log dir with timestamp
    import datetime
    timestamp = datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
    # Determine base log_dir
    base_log_dir = args.log_dir if args.log_dir is not None else train_config['log_dir'] + "_diffsim"
    log_dir = os.path.join(base_log_dir, timestamp)
    os.makedirs(log_dir, exist_ok=True)
    
    # Logger to save stdout to file
    class Logger(object):
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, "a")
            
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()
            
        def flush(self, *args, **kwargs):
            self.terminal.flush(*args, **kwargs)
            self.log.flush()
            
    sys.stdout = Logger(os.path.join(log_dir, "log.txt"))
    print(f"Logging to {log_dir}")
    # Config key 'seed' takes precedence over the --seed flag (default 0)
    seed = int(train_config.get('seed', args.seed))
    print(f"Random seed: {seed}")
    history_length = train_config['history_length']
    pos_loss_weight = float(train_config['pos_loss_weight'])
    force_loss_weight = float(train_config['force_loss_weight'])

    # Network & Optimization Params
    hidden_dim = int(train_config['hidden_dim'])
    latent_dim = int(train_config['latent_dim'])
    dropout_rate = float(train_config['dropout_rate'])
    weight_decay = float(train_config['weight_decay'])
    grad_clip = float(train_config['grad_clip'])
    # Stability options (default = legacy behavior)
    gripper_torque_clip = float(train_config.get('gripper_torque_clip', 5.0))
    qvel_clip = float(train_config.get('qvel_clip', 0.0))  # 0 = disabled
    normalize_features = bool(train_config.get('normalize_features', False))
    mask_invalid_force = bool(train_config.get('mask_invalid_force', False))
    gripper_loss_weight = float(train_config['gripper_loss_weight'])
    gate_loss_weight = float(train_config['gate_loss_weight'])
    condition_loss_weight = float(train_config.get('condition_loss_weight', 0.0))
    # Checkpoint saving strategy: "mae" (default) or "condition_accuracy"
    save_best_by = train_config.get('save_best_by', 'mae')
    if save_best_by not in ['mae', 'condition_accuracy']:
        raise ValueError(f"Invalid save_best_by value: {save_best_by}. Must be 'mae' or 'condition_accuracy'")
    print(f"Save best checkpoint by: {save_best_by}")
    # Model type selection (supports both new 'model_type' and legacy 'if_liquid_NN')
    model_type = get_model_type_from_config(train_config)
    print(f"Model type: {model_type}")

    # Focal weighting for force - higher weight for non-zero force samples
    force_focal_weight = float(train_config['force_focal_weight'])
    # Transition point of the smooth-L1 (Huber) force loss; 1.0 = classic smooth L1
    force_huber_beta = float(train_config.get('force_huber_beta', 1.0))
    if force_huber_beta != 1.0:
        print(f"Force huber beta: {force_huber_beta}")
    # Velocity-matching loss on the post-step sim qvel, supervised by finite
    # differences of the GT positions (NOT the CSV vel columns, whose telemetry
    # scale is off). 0 = disabled and leaves the loss graph unchanged.
    vel_loss_weight = float(train_config.get('vel_loss_weight', 0.0))
    if vel_loss_weight > 0:
        print(f"Velocity loss weight: {vel_loss_weight}")
    # Gate focal loss parameters (for enhanced gate learning)
    gate_focal_weight = float(train_config.get('gate_focal_weight', 1.0))  # gamma in focal loss
    gate_pos_weight = float(train_config.get('gate_pos_weight', 1.0))      # weight for positive class (contact)
    if gate_focal_weight != 1.0 or gate_pos_weight != 1.0:
        print(f"Gate focal loss enabled: gamma={gate_focal_weight}, pos_weight={gate_pos_weight}")
    sim_step_size = int(train_config['sim_step_size'])
    backbone_activation = train_config.get('backbone_activation', 'silu')  # For LNN CfC cell

    # Transformer specific parameters (optional for other models)
    num_heads = int(train_config.get('num_heads', 4))
    num_layers = int(train_config.get('num_layers', 2))
    d_ff = int(train_config.get('d_ff', 64))
    pool_type = train_config.get('pool_type', 'mean')
    use_gated_attention = train_config.get('use_gated_attention', False)

    # Initial Position Perturbation (Domain Randomization)
    # Add Gaussian noise to initial position to teach model to handle larger errors
    init_pos_noise_std = float(train_config['init_pos_noise_std'])
    if init_pos_noise_std > 0:
        print(f"Initial position perturbation enabled: std={init_pos_noise_std:.3f} rad ({np.rad2deg(init_pos_noise_std):.1f} deg)")
    else:
        print("Initial position perturbation disabled")

    # Goal Aperture Noise (Data Augmentation)
    # Add Gaussian noise to goal_aperture to prevent gripper overfitting
    # This forces the network to learn actual gripper dynamics instead of memorizing
    goal_aperture_noise_std = float(train_config.get('goal_aperture_noise_std', 0.0))
    if goal_aperture_noise_std > 0:
        print(f"Goal aperture noise enabled: std={goal_aperture_noise_std:.2f} mm")
    else:
        print("Goal aperture noise disabled")

    # Residual Torque Prediction Mode
    # When enabled, network predicts residual: final_torque = base_torque + residual
    # base_torque = current * torque_constant
    use_residual_torque = train_config['use_residual_torque']
    torque_constant = float(train_config['torque_constant'])  # Nm/A for XM430-W350
    if use_residual_torque:
        print(f"Residual torque mode enabled: final_torque = current * {torque_constant} Nm/A + network_output")
    else:
        print("Full torque mode: network directly predicts full torque")

    # Data timestep: actual dt between CSV rows
    base_data_dt = float(train_config['data_dt'])
    # Downsampling factor: take every N-th row from CSV
    downsample_factor = int(train_config['downsample_factor'])
    # Effective data_dt after downsampling
    data_dt = base_data_dt * downsample_factor
    if downsample_factor > 1:
        print(f"Downsampling enabled: factor={downsample_factor}, effective data_dt={data_dt:.4f}s ({1/data_dt:.1f}Hz)")
    
    # 2. Load MuJoCo Model
    mjcf_path = train_config.get('mjcf_path', 'robot/scene.xml')
    print(f"Loading MuJoCo model from {mjcf_path}...")
    abs_mjcf_path = os.path.abspath(mjcf_path)
    model_dir = os.path.dirname(abs_mjcf_path)
    cwd = os.getcwd()
    try:
        os.chdir(model_dir)
        mj_model = mujoco.MjModel.from_xml_path(os.path.basename(abs_mjcf_path))
    finally:
        os.chdir(cwd)
    
    # Enable x64 for MJX
    jax.config.update("jax_enable_x64", True)

    # Simulation timestep: data_dt / sim_step_size
    # data_dt = time between CSV rows (e.g., 0.016s for 62.5Hz data)
    # sim_step_size = number of MuJoCo steps per data point
    # Example: data_dt=0.016, sim_step_size=2 -> mj_timestep=0.008s
    sim_timestep = data_dt / sim_step_size
    mj_model.opt.timestep = sim_timestep
    print(f"Simulation setup: data_dt={data_dt:.4f}s ({1/data_dt:.1f}Hz), sim_step_size={sim_step_size}, mj_timestep={sim_timestep:.6f}s")
    
    # Solver settings for JAX differentiability
    mj_model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    mj_model.opt.iterations = 1
    mj_model.opt.ls_iterations = 0
    mj_model.opt.tolerance = 0
    mj_model.opt.ls_tolerance = 0
    mj_model.opt.noslip_iterations = 0
    mj_model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    
    mjx_model = mjx.put_model(mj_model)

    # Optionally apply the predicted external force as a real Cartesian force at the
    # grasp point (end_effector_target) during the rollout, so the interaction load
    # enters the dynamics through the force head instead of being absorbed by torque.
    apply_external_force = bool(train_config.get('apply_external_force', False))
    ee_target_id = int(mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, 'end_effector_target'))
    if apply_external_force:
        print(f"External force applied at grasp point 'end_effector_target' (id={ee_target_id})")

    # 3. Load Data
    csv_paths = []
    if args.csv:
        csv_paths.append(args.csv)
    elif 'datasets' in train_config:
        csv_paths = train_config['datasets']
    else:
        raise ValueError("No CSV data provided. Use --csv or define 'datasets' in config.")

    # Check for independent validation datasets (preferred over train/val split)
    val_csv_paths = train_config.get('val_datasets', [])
    use_independent_val = len(val_csv_paths) > 0
    if use_independent_val:
        print(f"Using independent validation datasets: {len(val_csv_paths)} files")
    else:
        print("No val_datasets in config, will use 90/10 train/val split")

    # =========================================================================
    # CRITICAL: Check CSV dt vs config data_dt alignment
    # =========================================================================
    print("\n" + "=" * 60)
    print("CHECKING CSV DATA RATE ALIGNMENT")
    print("=" * 60)
    for csv_path in csv_paths:
        if os.path.exists(csv_path):
            check_df = pd.read_csv(csv_path, nrows=100)  # Read first 100 rows for speed
            check_df.columns = [c.strip() for c in check_df.columns]
            if 'timestamp' in check_df.columns:
                csv_timestamps = check_df['timestamp'].values
                csv_dt_actual = np.mean(np.diff(csv_timestamps))
                csv_hz = 1.0 / csv_dt_actual
                config_hz = 1.0 / data_dt
                dt_error = abs(csv_dt_actual - data_dt)
                dt_error_pct = dt_error / data_dt * 100

                status = "OK" if dt_error < 0.002 else "MISMATCH!"
                print(f"  {os.path.basename(csv_path)}:")
                print(f"    CSV dt:    {csv_dt_actual:.6f}s ({csv_hz:.1f}Hz)")
                print(f"    Config dt: {data_dt:.6f}s ({config_hz:.1f}Hz)")
                print(f"    Error:     {dt_error:.6f}s ({dt_error_pct:.1f}%) {status}")

                if dt_error >= 0.002:
                    raise ValueError(
                        f"Data rate mismatch: CSV dt={csv_dt_actual:.4f}s but config data_dt={data_dt:.4f}s. "
                        f"Update data_dt in the config.")
            break  # Only check first CSV
    print("=" * 60 + "\n")
        
    # =========================================================================
    # Load data with trajectory boundary tracking
    # IMPORTANT: We load from original CSV files to preserve trajectory boundaries
    # Pre-merged split files (train.csv, val.csv) lose boundary info, so we skip them
    # =========================================================================
    print("\n" + "=" * 60)
    print("LOADING DATA WITH TRAJECTORY BOUNDARIES")
    print("=" * 60)

    # Load all datasets with boundary tracking
    result = load_dataset(csv_paths, mj_model, downsample_factor, return_boundaries=True, cfg=train_config)
    if result[0] is None:
        raise ValueError("No valid data loaded from csv_paths.")

    all_data_values, all_q_traj, all_v_traj, all_gt_pos, all_gt_force, all_force_valid, all_boundaries = result

    print(f"Loaded {len(all_boundaries)} trajectories, total {len(all_data_values)} samples")

    # =========================================================================
    # Create per-motor condition labels array based on file paths
    # Shape: (N, 5) - one condition per motor
    # Auto-detect: if path contains "degrade" -> joint3 (index 2) is degraded (0), others normal (1)
    # =========================================================================
    all_condition_gt = np.ones((len(all_data_values), 5), dtype=np.float32)  # Default: all motors normal

    if condition_loss_weight > 0:
        # Auto-detect from file paths: "degrade" in path means joint3 is degraded
        n_degraded_trajs = 0
        for traj_idx, (traj_start, traj_end) in enumerate(all_boundaries):
            csv_path = csv_paths[traj_idx]
            if "degrade" in csv_path.lower():
                # Only joint3 (index 2) is degraded, others remain normal
                all_condition_gt[traj_start:traj_end, 2] = 0.0  # Joint3 degraded
                n_degraded_trajs += 1
                print(f"  Traj {traj_idx}: Joint3=DEGRADED - {os.path.basename(csv_path)}")
            else:
                # All motors normal (already set to 1.0)
                print(f"  Traj {traj_idx}: All motors NORMAL - {os.path.basename(csv_path)}")

        n_normal_trajs = len(all_boundaries) - n_degraded_trajs
        print(f"Per-motor condition labels: {n_normal_trajs} normal trajs, {n_degraded_trajs} degraded trajs (only joint3)")
    else:
        # condition_loss_weight=0 - all normal (condition prediction disabled)
        print("Condition labels: all motors set to normal (1) - condition_loss_weight=0")

    # =========================================================================
    # Train/Val Split: Use independent val_datasets OR 90/10 split from train
    # =========================================================================
    if use_independent_val:
        # MODE 1: Use independent validation datasets (preferred)
        print("\n[MODE] Using independent validation datasets")

        # Train data = all loaded data (no split)
        train_data_values = all_data_values
        train_q_traj = all_q_traj
        train_v_traj = all_v_traj
        train_gt_pos = all_gt_pos
        train_gt_force = all_gt_force
        train_force_valid = all_force_valid
        train_cond_gt = all_condition_gt
        train_boundaries = all_boundaries

        print(f"Train: {len(train_data_values)} samples, {len(train_boundaries)} trajectories (100% of train data)")

        # Load validation data separately
        print(f"\nLoading validation datasets ({len(val_csv_paths)} files)...")
        val_result = load_dataset(val_csv_paths, mj_model, downsample_factor, return_boundaries=True, cfg=train_config)
        if val_result[0] is not None:
            val_data_values, val_q_traj, val_v_traj, val_gt_pos, val_gt_force, val_force_valid, val_boundaries = val_result
            # Create condition labels for val (all normal by default)
            val_cond_gt = np.ones((len(val_data_values), 5), dtype=np.float32)
            print(f"Val: {len(val_data_values)} samples, {len(val_boundaries)} trajectories")
        else:
            print("WARNING: Failed to load validation datasets, validation will be disabled")
            val_data_values = None
            val_q_traj = None
            val_v_traj = None
            val_gt_pos = None
            val_gt_force = None
            val_force_valid = None
            val_cond_gt = None
            val_boundaries = []
    else:
        # MODE 2: Split each trajectory into train/val (90/10) while preserving boundaries
        print("\n[MODE] Splitting train data 90/10 for train/val")
        train_split_ratio = 0.9
        train_data_list = []
        train_q_list = []
        train_v_list = []
        train_pos_list = []
        train_force_list = []
        train_force_valid_list = []
        train_cond_gt_list = []
        train_boundaries = []

        val_data_list = []
        val_q_list = []
        val_v_list = []
        val_pos_list = []
        val_force_list = []
        val_force_valid_list = []
        val_cond_gt_list = []
        val_boundaries = []

        train_offset = 0
        val_offset = 0

        for traj_idx, (traj_start, traj_end) in enumerate(all_boundaries):
            traj_len = traj_end - traj_start
            n_train = int(traj_len * train_split_ratio)
            n_val = traj_len - n_train

            # Extract trajectory data
            traj_data = all_data_values[traj_start:traj_end]
            traj_q = all_q_traj[traj_start:traj_end]
            traj_v = all_v_traj[traj_start:traj_end]
            traj_pos = all_gt_pos[traj_start:traj_end]
            traj_force = all_gt_force[traj_start:traj_end]
            traj_force_valid = all_force_valid[traj_start:traj_end]
            traj_cond_gt = all_condition_gt[traj_start:traj_end]

            # Train portion (first 90%)
            if n_train > 0:
                train_data_list.append(traj_data[:n_train])
                train_q_list.append(traj_q[:n_train])
                train_v_list.append(traj_v[:n_train])
                train_pos_list.append(traj_pos[:n_train])
                train_force_list.append(traj_force[:n_train])
                train_force_valid_list.append(traj_force_valid[:n_train])
                train_cond_gt_list.append(traj_cond_gt[:n_train])
                train_boundaries.append((train_offset, train_offset + n_train))
                train_offset += n_train

            # Val portion (last 10%)
            if n_val > 0:
                val_data_list.append(traj_data[n_train:])
                val_q_list.append(traj_q[n_train:])
                val_v_list.append(traj_v[n_train:])
                val_pos_list.append(traj_pos[n_train:])
                val_force_list.append(traj_force[n_train:])
                val_force_valid_list.append(traj_force_valid[n_train:])
                val_cond_gt_list.append(traj_cond_gt[n_train:])
                val_boundaries.append((val_offset, val_offset + n_val))
                val_offset += n_val

            print(f"  Traj {traj_idx}: len={traj_len} -> train={n_train}, val={n_val}")

        # Concatenate
        train_data_values = np.concatenate(train_data_list, axis=0) if train_data_list else None
        train_q_traj = np.concatenate(train_q_list, axis=0) if train_q_list else None
        train_v_traj = np.concatenate(train_v_list, axis=0) if train_v_list else None
        train_gt_pos = np.concatenate(train_pos_list, axis=0) if train_pos_list else None
        train_gt_force = np.concatenate(train_force_list, axis=0) if train_force_list else None
        train_force_valid = np.concatenate(train_force_valid_list, axis=0) if train_force_valid_list else None
        train_cond_gt = np.concatenate(train_cond_gt_list, axis=0) if train_cond_gt_list else None

        val_data_values = np.concatenate(val_data_list, axis=0) if val_data_list else None
        val_q_traj = np.concatenate(val_q_list, axis=0) if val_q_list else None
        val_v_traj = np.concatenate(val_v_list, axis=0) if val_v_list else None
        val_gt_pos = np.concatenate(val_pos_list, axis=0) if val_pos_list else None
        val_gt_force = np.concatenate(val_force_list, axis=0) if val_force_list else None
        val_force_valid = np.concatenate(val_force_valid_list, axis=0) if val_force_valid_list else None
        val_cond_gt = np.concatenate(val_cond_gt_list, axis=0) if val_cond_gt_list else None

        print(f"\nTrain: {len(train_data_values)} samples, {len(train_boundaries)} trajectory segments")
        if val_data_values is not None:
            print(f"Val: {len(val_data_values)} samples, {len(val_boundaries)} trajectory segments")

    # Print boundaries and condition distribution
    print(f"  Train boundaries: {train_boundaries[:5]}..." if len(train_boundaries) > 5 else f"  Train boundaries: {train_boundaries}")
    if val_boundaries:
        print(f"  Val boundaries: {val_boundaries[:5]}..." if len(val_boundaries) > 5 else f"  Val boundaries: {val_boundaries}")

    if train_cond_gt is not None:
        train_normal = int((train_cond_gt == 1.0).sum())
        train_degraded = int((train_cond_gt == 0.0).sum())
        print(f"  Train condition: {train_normal} normal, {train_degraded} degraded")
    if val_cond_gt is not None:
        val_normal = int((val_cond_gt == 1.0).sum())
        val_degraded = int((val_cond_gt == 0.0).sum())
        print(f"  Val condition: {val_normal} normal, {val_degraded} degraded")
    print("=" * 60 + "\n")

    if train_data_values is None:
        raise ValueError("No valid data loaded.")

    # =========================================================================
    # Validate joint limits BEFORE training (NO FALLBACKS)
    # =========================================================================
    if val_data_values is not None:
        validate_mujoco_joint_limits(mj_model, np.concatenate([train_data_values, val_data_values], axis=0))
    else:
        validate_mujoco_joint_limits(mj_model, train_data_values)

    # Feature normalization: per-feature z-score with training-set statistics.
    # Applied at the network input only; raw features are kept for the residual
    # torque (current in mA) and for the simulator. Stats are saved in checkpoints.
    global _NORM_STATS
    if normalize_features:
        feat_mean = train_data_values.mean(axis=0).astype(np.float32)
        feat_std = train_data_values.std(axis=0).astype(np.float32)
        # Per-channel std floors in native units. Columns that are near-constant in
        # the CSVs (e.g. aperture while holding an object) are overwritten by the
        # simulator during rollouts with a much wider range; a raw data std would
        # pin those channels to the clip rails and erase the feedback signal.
        std_floor = np.array(
            [0.05] * 5      # goal_pos1-5 (rad)
            + [0.05] * 4    # pos1-4 (rad)
            + [1.0]         # aperture (mm)
            + [10.0] * 5    # current1-5 (mA)
            + [0.1] * 5     # vel1-5 (rad/s)
            + [0.1] * 5     # volts1-5 (V)
            + [1.0] * 5     # temp1-5 (C)
            + [1.0]         # goal_aperture (mm)
            + [0.05] * 4    # arm error1-4 (rad)
            + [1.0],        # gripper_error (mm)
            dtype=np.float32)
        feat_std = np.maximum(feat_std, std_floor)
        _NORM_STATS = (feat_mean, feat_std)
        norm_mean_jax = jnp.array(feat_mean)
        norm_std_jax = jnp.array(feat_std)
        print(f"Feature normalization enabled (z-score, train stats). "
              f"std range: [{feat_std.min():.4g}, {feat_std.max():.4g}]")
    else:
        norm_mean_jax = None
        norm_std_jax = None

    def normalize_feat(x):
        if norm_mean_jax is None:
            return x
        return jnp.clip((x - norm_mean_jax) / norm_std_jax, -10.0, 10.0)

    # Convert Train to JAX
    gt_pos_jax = jnp.array(train_gt_pos)
    data_values_jax = jnp.array(train_data_values)
    gt_force_jax = jnp.array(train_gt_force)
    force_valid_jax = jnp.array(train_force_valid)
    cond_gt_jax = jnp.array(train_cond_gt)
    q_traj_jax = jnp.array(train_q_traj)
    v_traj_jax = jnp.array(train_v_traj)

    n_train_samples = len(train_data_values)
    feature_dim = train_data_values.shape[1] # Define feature_dim here
    print(f"Total loaded samples: {n_train_samples}")
    print(f"Train force valid ratio (mean over xyz channels): {train_force_valid.mean():.1%}")

    # Convert Val to JAX
    if val_data_values is not None:
        val_gt_pos_jax = jnp.array(val_gt_pos)
        val_data_values_jax = jnp.array(val_data_values)
        val_gt_force_jax = jnp.array(val_gt_force)
        val_force_valid_jax = jnp.array(val_force_valid)
        val_cond_gt_jax = jnp.array(val_cond_gt)
        n_val_samples = len(val_data_values)
        print(f"Val samples: {n_val_samples}")
        print(f"Val force valid ratio (mean over xyz channels): {val_force_valid.mean():.1%}")
    else:
        n_val_samples = 0
        val_force_valid_jax = None
        val_cond_gt_jax = None

    val_interval = train_config['val_interval']

    # Test set evaluation config (for early stopping)
    eval_interval = train_config['eval_interval']
    save_last_interval = train_config.get('save_last_interval', 100)  # Save last checkpoint every N epochs
    target_mae_threshold = train_config['target_mae_threshold']
    target_gripper_threshold = train_config.get('target_gripper_threshold', 1.0)  # mm
    target_condition_threshold = train_config.get('target_condition_threshold', 0.0)  # Condition accuracy threshold (0=disabled)
    target_force_threshold = train_config.get('target_force_threshold', 0.0)  # Force MAE threshold in N (0=disabled)
    test_datasets = train_config['test_datasets']
    train_eval_datasets = train_config.get('train_eval_datasets', {})
    test_condition_labels = train_config.get('test_condition_labels', {})  # Maps test dataset name -> condition (1=normal, 0=degraded)

    # Online learning mode: per-task early stopping (mean J1-J4 < 50% of table baseline)
    online_learning_mode = train_config.get('online_learning_mode', False)
    online_learning_window = train_config.get('online_learning_window', 600)  # Single window
    online_learning_target_reduction = train_config.get('online_learning_target_reduction', 0.5)
    online_learning_task_baselines = train_config.get('online_learning_task_baselines', {})  # Per-task baselines

    if eval_interval > 0 and test_datasets:
        print(f"Test set evaluation enabled: every {eval_interval} epochs")
        print(f"  Target threshold: all joints < {target_mae_threshold} degrees, gripper < {target_gripper_threshold} mm")
        if target_condition_threshold > 0:
            print(f"  Target condition accuracy: > {target_condition_threshold*100:.0f}%")
        if online_learning_mode:
            print(f"  [ONLINE LEARNING] Per-task early stopping @{online_learning_window} steps, target: {online_learning_target_reduction*100:.0f}% reduction")
            if online_learning_task_baselines:
                print(f"  Baselines for {len(online_learning_task_baselines)} tasks:")
        print(f"  Test datasets: {len(test_datasets)} tasks")
        if train_eval_datasets:
            print(f"  Train eval datasets: {len(train_eval_datasets)} tasks")

    # 4. Initialize Model
    # Note: data_dt is already loaded from config (default 0.016s)

    print(f"Initializing model: {model_type} (hidden_dim={hidden_dim}, latent_dim={latent_dim})")
    model = create_model(
        model_type=model_type,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        dropout_rate=dropout_rate,
        backbone_activation=backbone_activation,
        # Transformer specific
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        pool_type=pool_type,
        use_gated_attention=use_gated_attention,
        zero_init_head=bool(train_config.get('zero_init_torque_head', False)),
    )
    rng = jax.random.PRNGKey(seed)
    dummy_hist = jnp.ones((1, history_length, feature_dim))
    dummy_hist_flat = dummy_hist.reshape(1, -1)
    dummy_curr = jnp.ones((1, feature_dim))
    
    # Init with dropout key
    # All models have unified interface: (history, current, state, ts, training)
    # But state structure differs:
    # - MLP/Transformer: state=None (stateless)
    # - GRU/LNN: state=(h_torque, h_force)
    # - LSTM: state=((h_torque, c_torque), (h_force, c_force))
    rng, init_rng = jax.random.split(rng)
    dummy_h = jnp.zeros((1, hidden_dim))
    if model_type == 'lstm':
        # LSTM needs (h, c) for each path
        dummy_state = ((dummy_h, dummy_h), (dummy_h, dummy_h))
    elif model_type in ['gru', 'lnn']:
        # GRU/LNN need single h for each path
        dummy_state = (dummy_h, dummy_h)
    else:
        # MLP/Transformer are stateless
        dummy_state = None
    params = model.init({'params': init_rng, 'dropout': init_rng}, dummy_hist_flat, dummy_curr, dummy_state, ts=data_dt)

    # Print model size
    def count_params(params):
        return sum(x.size for x in jax.tree_util.tree_leaves(params))
    total_params = count_params(params)
    print(f"Model parameters: {total_params:,} ({total_params/1000:.1f}K)")

    # Load pretrained/resume checkpoint if provided (command line arg takes priority over config)
    resume_path = args.pretrained_path or train_config.get('resume_from', None)
    if resume_path:
        if os.path.exists(resume_path):
            print(f"Loading pretrained params from {resume_path}...")
            with open(resume_path, 'rb') as f:
                loaded_params = pickle.load(f)

            # Checkpoints may carry normalization stats alongside params.
            # A normalized checkpoint must keep ITS stats: recomputing them on new
            # data (or disabling normalization) would shift the input distribution
            # under weights that were trained on the original scale.
            if isinstance(loaded_params, dict) and ('feature_mean' in loaded_params or 'ema_params' in loaded_params):
                if 'feature_mean' in loaded_params:
                    if not normalize_features:
                        raise ValueError(
                            "Checkpoint was trained with feature normalization; "
                            "set normalize_features: true in the config to fine-tune it.")
                    ck_mean = np.asarray(loaded_params['feature_mean'], dtype=np.float32)
                    ck_std = np.asarray(loaded_params['feature_std'], dtype=np.float32)
                    _NORM_STATS = (ck_mean, ck_std)
                    norm_mean_jax = jnp.array(ck_mean)
                    norm_std_jax = jnp.array(ck_std)
                    print("  Adopted feature normalization stats from checkpoint (not recomputed)")
                loaded_params = loaded_params['params']
            elif normalize_features:
                print("  WARNING: fine-tuning a raw-input checkpoint with normalize_features=true; "
                      "the pretrained weights expect unnormalized inputs.")

            # Merge loaded params with initialized params (for partial checkpoint loading)
            # This handles:
            # 1. New keys (e.g., ConditionNet added) - keep random init
            # 2. Shape mismatch (e.g., feature_dim 34->36) - expand with random init
            def merge_params(init_params, loaded_params, prefix=""):
                """Recursively merge loaded params into init_params, handling shape mismatches."""
                if isinstance(init_params, dict):
                    merged = {}
                    for key in init_params:
                        full_key = f"{prefix}/{key}" if prefix else key
                        if key in loaded_params:
                            merged[key] = merge_params(init_params[key], loaded_params[key], full_key)
                        else:
                            print(f"    [NEW] {full_key} - keeping random initialization")
                            merged[key] = init_params[key]
                    # Check for keys in loaded but not in init (shouldn't happen normally)
                    for key in loaded_params:
                        if key not in init_params:
                            full_key = f"{prefix}/{key}" if prefix else key
                            print(f"    [SKIP] {full_key} - not in current model")
                    return merged
                else:
                    # Leaf node (actual parameter array)
                    init_shape = jnp.array(init_params).shape
                    loaded_shape = jnp.array(loaded_params).shape

                    if init_shape == loaded_shape:
                        # Shapes match - use loaded params
                        return loaded_params
                    elif len(init_shape) == len(loaded_shape):
                        # Same rank but different shape - try to expand
                        # This handles feature_dim expansion (e.g., 34->36)
                        init_arr = jnp.array(init_params)
                        loaded_arr = jnp.array(loaded_params)

                        # Check if we can expand (init >= loaded in all dims)
                        can_expand = all(i >= l for i, l in zip(init_shape, loaded_shape))
                        if can_expand:
                            # Copy loaded params into init params (keeping init for new dimensions)
                            result = init_arr  # Start with random init
                            # Build slice for loaded params
                            slices = tuple(slice(0, l) for l in loaded_shape)
                            result = result.at[slices].set(loaded_arr)
                            print(f"    [EXPAND] {prefix}: {loaded_shape} -> {init_shape}")
                            return result
                        else:
                            print(f"    [SHAPE MISMATCH] {prefix}: loaded={loaded_shape}, init={init_shape} - keeping init")
                            return init_params
                    else:
                        print(f"    [RANK MISMATCH] {prefix}: loaded={loaded_shape}, init={init_shape} - keeping init")
                        return init_params

            params = merge_params(params, loaded_params)

            # Debug: Check for NaN in merged params
            def check_nan_in_params(p, prefix=""):
                if isinstance(p, dict):
                    for k, v in p.items():
                        check_nan_in_params(v, f"{prefix}/{k}")
                else:
                    arr = jnp.array(p)
                    if jnp.isnan(arr).any():
                        print(f"    [NaN DETECTED] {prefix}")
                    if jnp.isinf(arr).any():
                        print(f"    [Inf DETECTED] {prefix}")
            check_nan_in_params(params)

            print(f"  Successfully loaded checkpoint (with partial merge if needed)!")
        else:
            print(f"WARNING: Resume path not found: {resume_path}")
            print(f"  Starting from random initialization...")

    # Optional LR schedule (one optimizer update per epoch)
    lr_warmup_epochs = int(train_config.get('lr_warmup_epochs', 0))
    if train_config.get('lr_schedule', '') == 'cosine':
        warmup_steps = min(max(lr_warmup_epochs, 1), max(epochs // 2, 1))
        # Decay horizon can be shorter/longer than the epoch budget; after
        # lr_decay_epochs the schedule holds the end value (lr * 0.1).
        lr_decay_epochs = int(train_config.get('lr_decay_epochs', epochs))
        lr_or_schedule = optax.warmup_cosine_decay_schedule(
            init_value=lr * 0.01, peak_value=lr,
            warmup_steps=warmup_steps,
            decay_steps=max(lr_decay_epochs, warmup_steps + 1), end_value=lr * 0.1)
        print(f"LR schedule: warmup {lr_warmup_epochs} epochs -> cosine decay to {lr*0.1:.2e} over {lr_decay_epochs} epochs")
    elif lr_warmup_epochs > 0:
        lr_or_schedule = optax.linear_schedule(init_value=lr * 0.01, end_value=lr,
                                               transition_steps=lr_warmup_epochs)
        print(f"LR warmup enabled: {lr_warmup_epochs} epochs ({lr*0.01:.2e} -> {lr:.2e})")
    else:
        lr_or_schedule = lr
    adam_b2 = float(train_config.get('adam_b2', 0.999))
    optimizer = optax.chain(
        optax.clip_by_global_norm(grad_clip),
        optax.adamw(lr_or_schedule, b2=adam_b2, weight_decay=weight_decay)
    )
    # Optionally skip updates with non-finite gradients instead of corrupting params
    if train_config.get('skip_nonfinite_updates', False):
        optimizer = optax.apply_if_finite(optimizer, max_consecutive_errors=200)
        print("Non-finite gradient updates will be skipped (apply_if_finite, max 200 consecutive)")
    state = TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

    # Optional EMA of parameters (evaluated/saved alongside raw params)
    global _EMA_PARAMS
    ema_decay = float(train_config.get('ema_decay', 0.0))
    if ema_decay > 0:
        _EMA_PARAMS = jax.tree_util.tree_map(jnp.array, state.params)
        _ema_update = jax.jit(lambda e, p: jax.tree_util.tree_map(
            lambda a, b: ema_decay * a + (1.0 - ema_decay) * b, e, p))
        print(f"EMA enabled: decay={ema_decay}")
    # Optionally track a separate best checkpoint selected by the EMA weights'
    # test score; raw-parameter selection is unaffected.
    eval_ema_params = bool(train_config.get('eval_ema_params', False)) and ema_decay > 0
    if eval_ema_params:
        print("EMA weights will also be scored on the test set (-> best_test_ema_params.pkl)")

    # 5. Define Rollout Loop
    
    def loss_fn(params, training, rng_keys, start_idx, batch_traj_starts, batch_gt_pos, batch_sensor_data, batch_gt_force, batch_force_valid, batch_cond_gt, batch_gt_vel=None):
        # rng_keys: (batch, steps, 2) or None if not training (but we need structure for scan)
        # batch_traj_starts: (batch,) - trajectory start indices for zero-padding calculation
        # batch_force_valid: (batch, steps, 3) - per channel: 1 where valid, 0 where -999
        # batch_cond_gt: (batch, steps) - motor condition labels (1=normal, 0=degraded)
        # batch_gt_vel: (batch, steps, 5) finite-difference GT velocity, only when vel_loss_weight > 0

        def step_fn(carry, inputs):
            # Unified carry structure: (mjx_data, history_buffer, step_idx, state)
            # state is (h_torque, h_force) for LNN, or dummy tuple for MLP
            mjx_data, history_buffer, step_idx, state = carry

            if vel_loss_weight > 0:
                target_pos, target_vel, csv_features, target_force, force_valid_step, cond_gt_step, rng_key = inputs
            else:
                target_pos, csv_features, target_force, force_valid_step, cond_gt_step, rng_key = inputs

            # 1. Construct Current Features (Hybrid)
            q = mjx_data.qpos
            v = mjx_data.qvel

            # 36D Feature Vector:
            # 0-4:   goal_pos1-5 (from CSV, CONTROL SIGNAL, unchanged)
            # 5-8:   pos1-4 (from simulation)
            # 9:     aperture (from simulation)
            # 10-14: current1-5 (from CSV, unchanged)
            # 15-19: vel1-5 (from simulation)
            # 20-24: volts1-5 (from CSV, unchanged)
            # 25-29: temp1-5 (from CSV, unchanged)
            # 30:    goal_aperture (from CSV, unchanged)
            # 31-34: error1-4 (goal_pos[:4] - pos[:4], computed from sim)
            # 35:    gripper_error (goal_aperture - aperture, computed from sim)
            current_feat = csv_features
            current_feat = current_feat.at[5:9].set(q[:4])    # pos1-4 from sim
            current_feat = current_feat.at[15:20].set(v[:5])  # vel1-5 from sim
            # volts (20-24), temp (25-29), goal_aperture (30) remain unchanged from CSV

            # Update Aperture from simulation
            # MuJoCo q[4] is single finger position (m), CSV aperture is also single finger (mm)
            aperture_val = q[4] * 1000.0
            current_feat = current_feat.at[9].set(aperture_val)  # aperture from sim

            # Apply goal_aperture noise during training (data augmentation)
            # This helps prevent gripper overfitting by forcing the network to learn dynamics
            if training and goal_aperture_noise_std > 0:
                rng_aperture, rng_key = jax.random.split(rng_key)
                aperture_noise = jax.random.normal(rng_aperture) * goal_aperture_noise_std
                noisy_goal_aperture = current_feat[30] + aperture_noise
                # Clip to valid gripper range [-11, 20] mm
                noisy_goal_aperture = jnp.clip(noisy_goal_aperture, -11.0, 20.0)
                current_feat = current_feat.at[30].set(noisy_goal_aperture)

            # Update error features from simulation
            # arm error: goal_pos[:4] - pos[:4] (indices 0-3 minus sim q[:4])
            arm_error = current_feat[0:4] - q[:4]
            current_feat = current_feat.at[31:35].set(arm_error)
            # gripper error: goal_aperture - aperture (index 30 minus sim aperture)
            # NOTE: If noise was applied, gripper_error will use the noisy goal_aperture
            gripper_error = current_feat[30] - aperture_val
            current_feat = current_feat.at[35].set(gripper_error)

            # 2. Predict Torque & Force (unified interface)
            hist_flat = history_buffer.reshape(-1)

            # RNG for dropout and gumbel
            if training:
                rng_dropout, rng_gumbel = jax.random.split(rng_key)
                rngs = {'dropout': rng_dropout, 'gumbel': rng_gumbel}
            else:
                rngs = None

            # Unified model.apply call - both MLP and LNN have same interface
            # Returns 6 values: torque, final_force, raw_force, gate, condition, new_state
            # The history buffer already lives in normalized space; normalize the
            # current frame the same way (identity when normalization is off).
            net_feat = normalize_feat(current_feat)
            tau_pred, final_force, raw_force, gate, condition, new_state = model.apply(
                params, hist_flat[None, :], net_feat[None, :], state,
                ts=data_dt, training=training, rngs=rngs
            )

            # Residual torque mode: final_torque = base_torque + network_output
            # base_torque = current * torque_constant
            # Current values are at indices 10-14 in csv_features (current1-5)
            # NOTE: Only apply residual to arm joints (0-3), gripper (4) uses direct prediction
            if use_residual_torque:
                current_values = csv_features[10:14]  # current1-4 in mA (arm only)
                # Convert mA to A, then multiply by torque constant
                base_torque = (current_values / 1000.0) * torque_constant
                # Arm: base_torque + residual, Gripper: direct prediction
                tau = jnp.concatenate([base_torque + tau_pred[0, :4], tau_pred[0, 4:5]])
            else:
                tau = tau_pred[0]  # (5,)

            f_pred = final_force[0] # (3,)
            gate_pred = gate[0, 0] # scalar
            cond_pred = condition[0]  # (5,) - per-motor condition (1=normal, 0=degraded)
            
            # 3. Step Simulation
            # Clamp torque to prevent simulation divergence from extreme predictions
            # XM430-W350 stall torque is ~4.1Nm, use +/-5Nm as conservative safety limit.
            # The gripper finger is much lighter than the arm links, so it gets its own
            # (typically tighter) limit via gripper_torque_clip.
            tau_limit = jnp.array([5.0, 5.0, 5.0, 5.0, gripper_torque_clip])
            tau_clamped = jnp.clip(tau, -tau_limit, tau_limit)
            ctrl = jnp.zeros(mjx_model.nu)
            ctrl = ctrl.at[:5].set(tau_clamped)

            mjx_data = mjx_data.replace(ctrl=ctrl)

            # Apply the predicted external force at the grasp point so the interaction
            # load acts through the dynamics (world-frame Cartesian force; MJX carries
            # the moment about each joint from the application point).
            if apply_external_force:
                xfrc = mjx_data.xfrc_applied.at[ee_target_id, :3].set(f_pred)
                mjx_data = mjx_data.replace(xfrc_applied=xfrc)

            # Step Simulation (Multi-step)
            def sim_loop_body(i, d):
                return mjx.step(mjx_model, d)

            mjx_data = jax.lax.fori_loop(0, sim_step_size, sim_loop_body, mjx_data)

            # Enforce gripper joint limits (MJCF slide-joint range [-0.011, 0.02] m).
            # MuJoCo doesn't auto-clamp, so we do it manually to prevent divergence.
            GRIPPER_MIN, GRIPPER_MAX = -0.011, 0.02
            clamped_qpos = mjx_data.qpos.at[4].set(jnp.clip(mjx_data.qpos[4], GRIPPER_MIN, GRIPPER_MAX))
            clamped_qpos = clamped_qpos.at[5].set(jnp.clip(clamped_qpos[5], GRIPPER_MIN, GRIPPER_MAX))

            # NaN protection: replace any NaN values with target position to prevent gradient corruption
            # This handles simulation divergence gracefully
            qpos_safe = jnp.nan_to_num(clamped_qpos, nan=0.0)
            # For the first 5 joints (controlled), use target as fallback for NaN
            nan_mask = jnp.isnan(clamped_qpos[:5])
            qpos_safe = qpos_safe.at[:5].set(jnp.where(nan_mask, target_pos, qpos_safe[:5]))
            mjx_data = mjx_data.replace(qpos=qpos_safe)

            # Optional qvel protection: joint velocities feed back into the network
            # features, so a diverging simulation can blow up training. Clip to a
            # physical bound (XM430 no-load speed ~4.8 rad/s) and scrub NaN/Inf.
            if qvel_clip > 0:
                qvel_safe = jnp.nan_to_num(
                    jnp.clip(mjx_data.qvel, -qvel_clip, qvel_clip), nan=0.0, posinf=qvel_clip, neginf=-qvel_clip)
                mjx_data = mjx_data.replace(qvel=qvel_safe)

            # 4. Update History (stored in normalized space; identity when off)
            new_hist = jnp.roll(history_buffer, -1, axis=0)
            new_hist = new_hist.at[-1].set(net_feat)

            # 5. Compute Loss & Metrics
            # IMPORTANT: Use q AFTER stepping, not before!
            q_after = mjx_data.qpos

            # Smooth L1 loss (Huber loss): less sensitive to outliers than MSE
            # smooth_l1(x) = 0.5 * x^2 if |x| < 1 else |x| - 0.5
            def smooth_l1(x):
                abs_x = jnp.abs(x)
                return jnp.where(abs_x < 1.0, 0.5 * x**2, abs_x - 0.5)

            # Split position error into arm (rad) and gripper (mm)
            # NOTE: Gripper error is converted to mm for comparable magnitude to arm joints (rad)
            # Without conversion: 5mm error -> 0.005m -> smooth_l1 ~ 0.0000125 (too small!)
            # With mm conversion: 5mm error -> 5.0 -> smooth_l1 ~ 4.5 (comparable to 5 deg arm error)
            arm_err = jnp.mean(smooth_l1(q_after[:4] - target_pos[:4]))  # Smooth L1 for 4 rotational joints
            grip_err = smooth_l1((q_after[4] - target_pos[4]) * 1000.0)  # Smooth L1 for gripper in mm

            # Velocity-matching loss (jitter suppression): compare the post-step
            # sim qvel (after the qvel clip above, i.e. exactly the state the
            # rollout carries forward) to the finite-difference GT velocity.
            # Same joint split as the position loss: arm joints in rad/s, the
            # gripper in mm/s weighted by gripper_loss_weight.
            if vel_loss_weight > 0:
                v_after = mjx_data.qvel
                arm_vel_err = jnp.mean(smooth_l1(v_after[:4] - target_vel[:4]))
                grip_vel_err = smooth_l1((v_after[4] - target_vel[4]) * 1000.0)
                vel_err = arm_vel_err + gripper_loss_weight * grip_vel_err

            # Force Loss with Focal Weighting
            # Non-zero force samples get higher weight to combat force imbalance
            # NOTE: -999 samples (converted to 0) are NOW supervised as force=0, gate=0
            # This teaches the network to predict "no contact" for these samples
            force_mag_gt = jnp.sqrt(jnp.sum(target_force**2))
            has_force = (force_mag_gt > 0.01).astype(jnp.float32)
            # Focal weight: force_focal_weight for non-zero, 1.0 for zero
            focal_weight = has_force * (force_focal_weight - 1.0) + 1.0
            # Use Smooth L1 instead of MSE to prevent gradient explosion
            # Use MEAN (not sum) for 3D force to prevent gradient explosion with xyz components
            # Default: supervise ALL samples including -999 (no contact = 0).
            # With mask_invalid_force, sentinel channels drop out of the force
            # loss (the gate is still supervised to 0 there via gate_gt).
            # The force term can use its own Huber transition point (beta);
            # at beta = 1.0 this is exactly smooth_l1 above.
            def smooth_l1_force(x):
                if force_huber_beta == 1.0:
                    return smooth_l1(x)
                abs_x = jnp.abs(x)
                return jnp.where(abs_x < force_huber_beta,
                                 0.5 * x**2 / force_huber_beta,
                                 abs_x - 0.5 * force_huber_beta)
            if mask_invalid_force:
                force_err_raw = jnp.mean(smooth_l1_force(f_pred - target_force) * force_valid_step)
            else:
                force_err_raw = jnp.mean(smooth_l1_force(f_pred - target_force))
            force_err = force_err_raw * focal_weight

            # Gate Loss (BCE with Focal Loss and Class Weighting)
            # GT Gate: 1 if |force| > 0.01, else 0
            # For -999 samples (force=0), gate_gt=0, teaching network to predict "no contact"
            force_mag = force_mag_gt
            gate_gt = has_force

            # Focal Loss: - alpha * (1-p)^gamma * log(p) for positive, - (1-alpha) * p^gamma * log(1-p) for negative
            # With class weighting: pos_weight for contact class to handle imbalance
            gate_pred_clipped = jnp.clip(gate_pred, 1e-7, 1.0 - 1e-7)
            
            # Focal modulation: (1-p_t)^gamma where p_t is predicted prob for true class
            p_t = gate_gt * gate_pred_clipped + (1.0 - gate_gt) * (1.0 - gate_pred_clipped)
            focal_modulation = jnp.power(1.0 - p_t, gate_focal_weight)
            
            # Class-weighted BCE with focal modulation
            # Positive class (contact, gate_gt=1): weight = gate_pos_weight
            # Negative class (no contact, gate_gt=0): weight = 1.0
            class_weight = gate_gt * gate_pos_weight + (1.0 - gate_gt) * 1.0
            
            bce = - (gate_gt * jnp.log(gate_pred_clipped) + (1.0 - gate_gt) * jnp.log(1.0 - gate_pred_clipped))
            gate_err_raw = focal_modulation * class_weight * bce
            # No masking - supervise ALL samples including -999 (no contact = gate=0)
            gate_err = gate_err_raw

            # Joint3-Only Condition Loss (BCE)
            # GT Condition: (5,) - 1 = normal motor, 0 = degraded/damaged motor
            # Only joint3 (index 2) can be degraded, others are always normal
            # So we only compute condition loss for joint3 to avoid diluting the gradient
            cond_gt = cond_gt_step  # (5,)
            cond_pred_clipped = jnp.clip(cond_pred, 1e-7, 1.0 - 1e-7)  # (5,)
            cond_err_per_motor = - (cond_gt * jnp.log(cond_pred_clipped) + (1.0 - cond_gt) * jnp.log(1.0 - cond_pred_clipped))
            cond_err = cond_err_per_motor[2]  # Only joint3 (index 2) - the only motor that can be degraded

            pos_mae = jnp.mean(jnp.abs(q_after[:5] - target_pos))
            force_mae = jnp.mean(jnp.abs(f_pred - target_force))

            # Per-joint MAE
            diff = jnp.abs(q_after[:5] - target_pos)
            mae_j1_deg = diff[0] * 180.0 / jnp.pi
            mae_j2_deg = diff[1] * 180.0 / jnp.pi
            mae_j3_deg = diff[2] * 180.0 / jnp.pi
            mae_j4_deg = diff[3] * 180.0 / jnp.pi
            mae_grip_mm = diff[4] * 1000.0

            per_joint_mae = jnp.array([mae_j1_deg, mae_j2_deg, mae_j3_deg, mae_j4_deg, mae_grip_mm])

            # Gate Accuracy (for monitoring)
            gate_acc = ((gate_pred > 0.5) == (gate_gt > 0.5)).astype(jnp.float32)

            # Joint3-Only Condition Classification Metrics (for paper reporting)
            # Positive class = degraded (cond=0), Negative class = normal (cond=1)
            # Only joint3 (index 2) matters - other motors are always normal
            cond_pred_j3 = cond_pred[2]  # scalar - joint3 prediction
            cond_gt_j3 = cond_gt[2]      # scalar - joint3 ground truth
            cond_pred_binary = (cond_pred_j3 < 0.5).astype(jnp.float32)  # 1 if predicted degraded
            cond_gt_binary = (cond_gt_j3 < 0.5).astype(jnp.float32)      # 1 if actual degraded
            # TP/TN/FP/FN for joint3 only
            cond_tp = cond_pred_binary * cond_gt_binary               # Both degraded
            cond_tn = (1 - cond_pred_binary) * (1 - cond_gt_binary)   # Both normal
            cond_fp = cond_pred_binary * (1 - cond_gt_binary)         # Predicted degraded, actual normal
            cond_fn = (1 - cond_pred_binary) * cond_gt_binary         # Predicted normal, actual degraded

            # Unified return - new_state is tuple for LNN, None for MLP (but we keep tuple for consistency)
            # For MLP, new_state is None, but we need consistent carry structure for jax.lax.scan
            # So we pass the same state through (it won't be used by MLP anyway)
            next_state = new_state if new_state is not None else state
            step_out = (arm_err, grip_err, force_err, gate_err, cond_err, pos_mae, force_mae, per_joint_mae, gate_acc, has_force, tau, cond_tp, cond_tn, cond_fp, cond_fn, cond_pred_j3, cond_gt_j3)
            if vel_loss_weight > 0:
                step_out = step_out + (vel_err,)
            return (mjx_data, new_hist, step_idx + 1, next_state), step_out

        # Vmap over batch
        def rollout_single(start_i, traj_start_i, gt_pos_seq, sensor_seq, gt_force_seq, force_valid_seq, cond_gt_seq, rng_seq, gt_vel_seq=None):
            """Rollout single trajectory with zero-padding support for history buffer.

            Args:
                start_i: Start index in global data array
                traj_start_i: Start index of current trajectory (for zero-padding boundary)
                cond_gt_seq: Motor condition labels for each step (1=normal, 0=degraded)
            """
            init_q = q_traj_jax[start_i]
            init_v = v_traj_jax[start_i]

            # Initial Position Perturbation (Domain Randomization)
            # Add Gaussian noise to initial position to teach model to handle larger errors
            # Use first key from rng_seq for perturbation
            if init_pos_noise_std > 0:
                perturb_key = rng_seq[0]  # Use first step's key for perturbation
                # Only perturb the first 4 joints (arm joints), not the gripper (joint 5)
                noise = jax.random.normal(perturb_key, shape=(4,)) * init_pos_noise_std
                # Pad with zero for gripper (no perturbation on gripper)
                noise_padded = jnp.concatenate([noise, jnp.zeros(init_q.shape[0] - 4)])
                init_q = init_q + noise_padded

            mjx_data = mjx.make_data(mjx_model)
            mjx_data = mjx_data.replace(qpos=init_q, qvel=init_v)

            # Zero-padding for history buffer when near trajectory start
            # traj_start_i is the start index of current trajectory
            hist_start = start_i - jnp.int32(history_length)

            # Create indices for history buffer: [hist_start, hist_start+1, ..., start_i-1]
            hist_indices = hist_start + jnp.arange(history_length, dtype=jnp.int32)

            # Valid mask: indices must be >= traj_start (within current trajectory)
            valid_mask = hist_indices >= traj_start_i  # shape: (history_length,)

            # Safe indices: clamp to valid range to avoid out-of-bounds access
            safe_indices = jnp.maximum(hist_indices, traj_start_i)

            # Gather data using advanced indexing
            hist_data = data_values_jax[safe_indices]  # shape: (history_length, feature_dim)

            # Apply zero-padding mask: invalid positions become zeros.
            # With normalization on, frames are normalized first, so the zero pad
            # corresponds to the training-set mean frame (a neutral placeholder).
            hist_buf = jnp.where(valid_mask[:, None], normalize_feat(hist_data), jnp.zeros_like(hist_data))

            # Unified state initialization based on model type
            # - MLP/Transformer: state=None (stateless)
            # - GRU/LNN: state=(h_torque, h_force)
            # - LSTM: state=((h_torque, c_torque), (h_force, c_force))
            if model_type == 'lstm':
                # LSTM has separate h and c states
                if 'h0_torque' in params['params']:
                    h0_torque = params['params']['h0_torque']
                    c0_torque = params['params']['c0_torque']
                    h0_force = params['params']['h0_force']
                    c0_force = params['params']['c0_force']
                else:
                    h0_torque = jnp.zeros((1, hidden_dim))
                    c0_torque = jnp.zeros((1, hidden_dim))
                    h0_force = jnp.zeros((1, hidden_dim))
                    c0_force = jnp.zeros((1, hidden_dim))
                init_state = ((h0_torque, c0_torque), (h0_force, c0_force))
            elif model_type in ['gru', 'lnn']:
                # GRU/LNN have single h state per path
                if 'h0_torque' in params['params']:
                    h0_torque = params['params']['h0_torque']
                    h0_force = params['params']['h0_force']
                else:
                    h0_torque = jnp.zeros((1, hidden_dim))
                    h0_force = jnp.zeros((1, hidden_dim))
                init_state = (h0_torque, h0_force)
            else:
                # MLP/Transformer are stateless
                init_state = None
            init_carry = (mjx_data, hist_buf, 0, init_state)

            if vel_loss_weight > 0:
                scan_inputs = (gt_pos_seq, gt_vel_seq, sensor_seq, gt_force_seq, force_valid_seq, cond_gt_seq, rng_seq)
            else:
                scan_inputs = (gt_pos_seq, sensor_seq, gt_force_seq, force_valid_seq, cond_gt_seq, rng_seq)
            final_carry, scan_out = jax.lax.scan(
                step_fn,
                init_carry,
                scan_inputs
            )
            if vel_loss_weight > 0:
                vel_losses = scan_out[-1]
                scan_out = scan_out[:-1]
            (arm_losses, grip_losses, force_losses, gate_losses, cond_losses, pos_maes, force_maes, per_joint_maes, gate_accs, has_forces, taus, cond_tps, cond_tns, cond_fps, cond_fns, cond_preds, cond_gts) = scan_out

            # Compute tau statistics for debugging mode collapse
            # taus shape: (rollout_steps, 5)
            tau_mean = jnp.mean(taus, axis=0)  # (5,) - mean per joint
            tau_std = jnp.std(taus, axis=0)    # (5,) - std per joint
            tau_min = jnp.min(taus, axis=0)   # (5,) - min per joint
            tau_max = jnp.max(taus, axis=0)   # (5,) - max per joint

            # Sum condition metrics (counts, not means) for precision/recall calculation
            cond_tp_sum = jnp.sum(cond_tps)
            cond_tn_sum = jnp.sum(cond_tns)
            cond_fp_sum = jnp.sum(cond_fps)
            cond_fn_sum = jnp.sum(cond_fns)

            rollout_out = (jnp.mean(arm_losses), jnp.mean(grip_losses), jnp.mean(force_losses), jnp.mean(gate_losses), jnp.mean(cond_losses),
                    jnp.mean(pos_maes), jnp.mean(force_maes), jnp.mean(per_joint_maes, axis=0), jnp.mean(gate_accs),
                    jnp.mean(has_forces), tau_mean, tau_std, tau_min, tau_max,
                    cond_tp_sum, cond_tn_sum, cond_fp_sum, cond_fn_sum,
                    cond_preds, cond_gts)  # (rollout_steps,) each - joint3 only, for AUC calculation
            if vel_loss_weight > 0:
                rollout_out = rollout_out + (jnp.mean(vel_losses),)
            return rollout_out

        if vel_loss_weight > 0:
            vmap_out = jax.vmap(rollout_single)(
                start_idx, batch_traj_starts, batch_gt_pos, batch_sensor_data, batch_gt_force, batch_force_valid, batch_cond_gt, rng_keys, batch_gt_vel
            )
            batch_vel_loss = vmap_out[-1]
            vmap_out = vmap_out[:-1]
        else:
            vmap_out = jax.vmap(rollout_single)(
                start_idx, batch_traj_starts, batch_gt_pos, batch_sensor_data, batch_gt_force, batch_force_valid, batch_cond_gt, rng_keys
            )
        (batch_arm_loss, batch_grip_loss, batch_force_loss, batch_gate_loss, batch_cond_loss, batch_pos_mae, batch_force_mae,
         batch_per_joint_mae, batch_gate_acc, batch_has_force,
         batch_tau_mean, batch_tau_std, batch_tau_min, batch_tau_max,
         batch_cond_tp, batch_cond_tn, batch_cond_fp, batch_cond_fn,
         batch_cond_preds, batch_cond_gts) = vmap_out
        # batch_cond_preds/gts shape: (batch, rollout_steps) - joint3 only

        total_arm_loss = jnp.mean(batch_arm_loss)
        total_grip_loss = jnp.mean(batch_grip_loss)
        total_force_loss = jnp.mean(batch_force_loss)
        total_gate_loss = jnp.mean(batch_gate_loss)
        total_cond_loss = jnp.mean(batch_cond_loss)

        total_pos_mae = jnp.mean(batch_pos_mae)
        total_force_mae = jnp.mean(batch_force_mae)
        total_per_joint_mae = jnp.mean(batch_per_joint_mae, axis=0)
        total_gate_acc = jnp.mean(batch_gate_acc)
        total_has_force_ratio = jnp.mean(batch_has_force)  # Ratio of non-zero force samples

        # Aggregate tau statistics across batch
        # batch_tau_* shapes: (batch, 5)
        total_tau_mean = jnp.mean(batch_tau_mean, axis=0)  # (5,) mean of means
        total_tau_std = jnp.mean(batch_tau_std, axis=0)    # (5,) mean of stds (within-rollout variance)
        total_tau_min = jnp.min(batch_tau_min, axis=0)     # (5,) global min
        total_tau_max = jnp.max(batch_tau_max, axis=0)     # (5,) global max

        # Aggregate condition classification metrics (sum across batch)
        total_cond_tp = jnp.sum(batch_cond_tp)
        total_cond_tn = jnp.sum(batch_cond_tn)
        total_cond_fp = jnp.sum(batch_cond_fp)
        total_cond_fn = jnp.sum(batch_cond_fn)

        # Calculate Precision, Recall, F1, Accuracy for condition classification
        # Positive class = degraded motor (cond=0)
        cond_precision = total_cond_tp / (total_cond_tp + total_cond_fp + 1e-7)
        cond_recall = total_cond_tp / (total_cond_tp + total_cond_fn + 1e-7)  # = TPR = Sensitivity
        cond_f1 = 2 * cond_precision * cond_recall / (cond_precision + cond_recall + 1e-7)
        cond_accuracy = (total_cond_tp + total_cond_tn) / (total_cond_tp + total_cond_tn + total_cond_fp + total_cond_fn + 1e-7)

        # Additional metrics for paper reporting
        cond_specificity = total_cond_tn / (total_cond_tn + total_cond_fp + 1e-7)  # TNR
        cond_balanced_acc = (cond_recall + cond_specificity) / 2.0  # (TPR + TNR) / 2
        cond_fnr = total_cond_fn / (total_cond_fn + total_cond_tp + 1e-7)  # Miss rate
        cond_fpr = total_cond_fp / (total_cond_fp + total_cond_tn + 1e-7)  # False alarm rate

        # AUC-ROC calculation using Wilcoxon-Mann-Whitney statistic
        # Flatten predictions and labels across batch and steps
        all_preds = batch_cond_preds.flatten()  # (batch * rollout_steps,)
        all_gts = batch_cond_gts.flatten()      # (batch * rollout_steps,)

        # Positive class = degraded (gt < 0.5), Negative class = normal (gt >= 0.5)
        # For degraded samples, we expect low cond_pred; for normal, high cond_pred
        # AUC = P(pred_normal > pred_degraded) for all (normal, degraded) pairs
        pos_mask = all_gts < 0.5   # degraded samples
        neg_mask = all_gts >= 0.5  # normal samples
        n_pos = jnp.sum(pos_mask)
        n_neg = jnp.sum(neg_mask)

        # Compute AUC using rank-based method (efficient approximation)
        # Sort predictions and compute ranks
        # For each positive sample, count how many negative samples have higher pred
        # This equals AUC * n_pos * n_neg
        def compute_auc_wmw(preds, pos_mask, neg_mask, n_pos, n_neg):
            """Compute AUC using Wilcoxon-Mann-Whitney statistic."""
            # Positive class (degraded) should get a LOW cond_pred, so rank on 1 - pred:
            # AUC = (R_pos - n_pos*(n_pos+1)/2) / (n_pos * n_neg)
            flipped_preds = 1.0 - preds
            sorted_indices_flip = jnp.argsort(flipped_preds)
            ranks_flip = jnp.zeros_like(flipped_preds)
            ranks_flip = ranks_flip.at[sorted_indices_flip].set(jnp.arange(1, len(flipped_preds) + 1, dtype=flipped_preds.dtype))

            pos_rank_sum = jnp.sum(jnp.where(pos_mask, ranks_flip, 0.0))
            auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg + 1e-7)
            return jnp.clip(auc, 0.0, 1.0)

        # Only compute AUC if we have both classes
        has_both_classes = (n_pos > 0) & (n_neg > 0)
        cond_auc = jnp.where(
            has_both_classes,
            compute_auc_wmw(all_preds, pos_mask, neg_mask, n_pos, n_neg),
            0.5  # Random guess when only one class present
        )

        # Fixed weights loss
        total_pos_loss = total_arm_loss + gripper_loss_weight * total_grip_loss
        total_loss = (pos_loss_weight * total_pos_loss +
                     force_loss_weight * total_force_loss +
                     gate_loss_weight * total_gate_loss +
                     condition_loss_weight * total_cond_loss)
        if vel_loss_weight > 0:
            total_vel_loss = jnp.mean(batch_vel_loss)
            total_loss = total_loss + vel_loss_weight * total_vel_loss
        w_arm, w_grip, w_force, w_gate, w_cond = pos_loss_weight, gripper_loss_weight, force_loss_weight, gate_loss_weight, condition_loss_weight

        # Return detailed metrics including learned weights and tau stats
        aux = (total_arm_loss, total_grip_loss, total_force_loss, total_gate_loss, total_cond_loss,
                           total_pos_mae, total_force_mae, total_per_joint_mae, total_gate_acc,
                           w_arm, w_grip, w_force, w_gate, w_cond, total_has_force_ratio,
                           total_tau_mean, total_tau_std, total_tau_min, total_tau_max,
                           cond_accuracy, cond_precision, cond_recall, cond_f1,
                           cond_specificity, cond_balanced_acc, cond_fnr, cond_fpr, cond_auc)
        if vel_loss_weight > 0:
            aux = aux + (total_vel_loss,)
        return total_loss, aux

    @jax.jit
    def train_step(state, rng, start_idx, batch_traj_starts, batch_gt_pos, batch_sensor_data, batch_gt_force, batch_force_valid, batch_cond_gt, batch_gt_vel=None):
        # Generate RNG keys for dropout
        # We need (batch, steps, 2) keys for the full batch rollout
        batch_size_local = batch_gt_pos.shape[0]
        steps = batch_gt_pos.shape[1]

        # Split into batch keys, then each into step keys
        batch_keys = jax.random.split(rng, batch_size_local)
        # (batch, steps, 2)
        rng_keys = jax.vmap(lambda k: jax.random.split(k, steps))(batch_keys)

        def loss_wrapper(params):
            return loss_fn(params, True, rng_keys, start_idx, batch_traj_starts,
                          batch_gt_pos, batch_sensor_data, batch_gt_force, batch_force_valid, batch_cond_gt,
                          batch_gt_vel)

        (loss, aux), grads = jax.value_and_grad(loss_wrapper, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)

        return state, loss, aux

    @jax.jit
    def validate_step(state, rng, start_idx, batch_traj_starts, batch_gt_pos, batch_sensor_data, batch_gt_force, batch_force_valid, batch_cond_gt, batch_gt_vel=None):
        # Generate RNG keys for dropout (even though training=False, we need the structure)
        batch_size_local = batch_gt_pos.shape[0]
        steps = batch_gt_pos.shape[1]

        batch_keys = jax.random.split(rng, batch_size_local)
        rng_keys = jax.vmap(lambda k: jax.random.split(k, steps))(batch_keys)

        loss, aux = loss_fn(state.params, False, rng_keys, start_idx, batch_traj_starts,
                           batch_gt_pos, batch_sensor_data, batch_gt_force, batch_force_valid, batch_cond_gt,
                           batch_gt_vel)

        return loss, aux

    # 6. Training Loop
    writer = SummaryWriter(log_dir)
    print("Starting DiffSim training...")
    
    # =========================================================================
    # Validate trajectory boundaries for sampling
    # =========================================================================
    print("Validating trajectory boundaries for sampling...")

    # Create numpy RNG for reproducible sampling
    np_rng = np.random.default_rng(seed=42 + seed)

    # Test sampling with debug output
    print("\n[DEBUG] Testing train sampling:")
    _ = sample_valid_indices(train_boundaries, history_length, rollout_steps, batch_size, rng=np_rng, debug=True)

    if n_val_samples > 0 and val_boundaries:
        print("\n[DEBUG] Testing val sampling:")
        _ = sample_valid_indices(val_boundaries, history_length, rollout_steps, batch_size, rng=np_rng, debug=True)

    # Reset RNG for actual training
    np_rng = np.random.default_rng(seed=seed)

    min_val_loss = float('inf')
    min_train_loss = float('inf')
    best_test_mae = float('inf')  # Best test set MAE @Full trajectory (max of J1-J4)
    best_test_ema_mae = float('inf')  # Best test MAE achieved by the EMA weights
    best_test_condition_acc = 0.0  # Best test set condition accuracy (for motor condition task)
    # Initialize validation metrics (for printing when val hasn't run yet)
    val_loss = None
    val_mae_pos = None
    val_mae_force = None
    
    # Training time tracking (for online learning statistics)
    training_start_time = time.time()
    epoch_times = []  # Track time per epoch for statistics
    total_eval_time = 0.0  # Track cumulative eval time (to exclude from training time)

    pbar = tqdm(range(epochs), desc="Training", ncols=120)
    # Optional rollout-length curriculum (paper appendix: 128 -> 256 -> final length).
    # Changing the window length changes batch shapes, so JAX retraces automatically
    # at each stage transition (a few extra compiles over the whole run).
    curriculum_epochs = train_config.get('curriculum_epochs', None)  # e.g. [2000, 5000]
    curriculum_steps = train_config.get('curriculum_steps', None)    # e.g. [128, 256]
    # Alternative: sample the rollout length per epoch from a small fixed set
    # (small set keeps the number of JIT traces bounded).
    rollout_choices = train_config.get('rollout_length_choices', None)

    def rollout_for_epoch(ep):
        if rollout_choices:
            return int(np_rng.choice(rollout_choices))
        if not curriculum_epochs:
            return rollout_steps
        for bound, steps in zip(curriculum_epochs, curriculum_steps):
            if ep < bound:
                return int(steps)
        return rollout_steps

    for epoch in pbar:
        cur_rollout = rollout_for_epoch(epoch)
        # Sample batch using boundary-aware sampling (returns start_indices AND traj_starts for zero-padding)
        start_indices, traj_starts = sample_valid_indices(
            train_boundaries, history_length, cur_rollout, batch_size,
            rng=np_rng, debug=(epoch == 0)  # Debug output on first epoch only
        )

        # Prepare batches
        batch_gt_pos = []
        batch_sensor_data = []
        batch_gt_force = []
        batch_force_valid = []
        batch_cond_gt = []
        batch_gt_vel = [] if vel_loss_weight > 0 else None

        for idx in start_indices:
            # Target is shifted by 1: we predict torque at t to reach state at t+1
            # init_q = GT[idx], target = GT[idx+1:idx+1+cur_rollout]
            batch_gt_pos.append(gt_pos_jax[idx+1:idx+1+cur_rollout])
            batch_sensor_data.append(data_values_jax[idx:idx+cur_rollout])
            batch_gt_force.append(gt_force_jax[idx+1:idx+1+cur_rollout])
            batch_force_valid.append(force_valid_jax[idx+1:idx+1+cur_rollout])
            batch_cond_gt.append(cond_gt_jax[idx+1:idx+1+cur_rollout])
            if vel_loss_weight > 0:
                # GT velocity by finite difference, aligned with the shifted
                # targets: vel target at step t is (GT[idx+1+t] - GT[idx+t]) / data_dt
                batch_gt_vel.append((gt_pos_jax[idx+1:idx+1+cur_rollout] - gt_pos_jax[idx:idx+cur_rollout]) / data_dt)

        batch_gt_pos = jnp.array(batch_gt_pos)
        batch_sensor_data = jnp.array(batch_sensor_data)
        batch_gt_force = jnp.array(batch_gt_force)
        batch_force_valid = jnp.array(batch_force_valid)
        batch_cond_gt = jnp.array(batch_cond_gt)
        if vel_loss_weight > 0:
            batch_gt_vel = jnp.array(batch_gt_vel)
        start_indices_jax = jnp.array(start_indices)
        batch_traj_starts = jnp.array(traj_starts)

        t0 = time.time()

        # Split RNG for this step
        rng, step_rng = jax.random.split(rng)

        state, loss, loss_comps = train_step(state, step_rng, start_indices_jax, batch_traj_starts, batch_gt_pos, batch_sensor_data, batch_gt_force, batch_force_valid, batch_cond_gt, batch_gt_vel)
        if ema_decay > 0:
            _EMA_PARAMS = _ema_update(_EMA_PARAMS, state.params)
        t1 = time.time()
        
        # NaN Loss Detection
        # With skip_nonfinite_updates the parameters stay untouched on a bad batch
        # (apply_if_finite skipped the update), so tolerate isolated NaN batches and
        # only abort after many in a row. Without the option, keep the hard exit.
        if np.isnan(float(loss)):
            nan_epoch_streak = getattr(main, '_nan_streak', 0) + 1
            main._nan_streak = nan_epoch_streak
            if train_config.get('skip_nonfinite_updates', False) and nan_epoch_streak < 50:
                print(f"[WARN] NaN loss at epoch {epoch} (streak {nan_epoch_streak}/50); update skipped, continuing.")
                continue
            print(f"\n" + "="*60)
            print(f"FATAL ERROR: Loss is NaN at epoch {epoch}!")
            print(f"="*60)
            print(f"Training terminated immediately to prevent corrupted checkpoints.")
            print(f"Possible causes:")
            print(f"  1. Learning rate too high")
            print(f"  2. Gradient explosion")
            print(f"  3. Numerical instability in simulation")
            print(f"  4. Bad data (NaN/Inf in input)")
            print(f"="*60)
            writer.close()
            sys.exit(1)
        else:
            main._nan_streak = 0
        
        # Unpack train loss components
        # aux_mean structure: (arm_loss, grip_loss, force_loss, gate_loss, cond_loss, pos_mae, force_mae, per_joint_mae, gate_acc,
        #                      w_arm, w_grip, w_force, w_gate, w_cond, has_force_ratio, tau_mean, tau_std, tau_min, tau_max,
        #                      cond_accuracy, cond_precision, cond_recall, cond_f1,
        #                      cond_specificity, cond_balanced_acc, cond_fnr, cond_fpr, cond_auc)
        loss_vel = None
        if vel_loss_weight > 0:
            loss_vel = loss_comps[-1]
            loss_comps = loss_comps[:-1]
        (loss_arm, loss_grip, loss_force, loss_gate, loss_cond, mae_pos, mae_force, per_joint_mae, acc_gate,
         w_arm, w_grip, w_force, w_gate, w_cond, has_force_ratio,
         tau_mean, tau_std, tau_min, tau_max,
         cond_acc, cond_prec, cond_rec, cond_f1,
         cond_spec, cond_bal_acc, cond_fnr, cond_fpr, cond_auc) = loss_comps

        # Combined position loss for backwards compatibility
        loss_pos = loss_arm + loss_grip

        # Update tqdm progress bar
        pbar.set_postfix({
            'loss': f'{loss:.4f}',
            'J1': f'{per_joint_mae[0]:.1f}°',
            'J2': f'{per_joint_mae[1]:.1f}°',
            'J3': f'{per_joint_mae[2]:.1f}°',
            'J4': f'{per_joint_mae[3]:.1f}°',
            'G': f'{per_joint_mae[4]:.1f}mm'
        })

        # DEBUG: Print tau statistics every 100 epochs to check for mode collapse
        if epoch % 100 == 0:
            tau_mean_np = np.array(tau_mean)
            tau_std_np = np.array(tau_std)
            tau_min_np = np.array(tau_min)
            tau_max_np = np.array(tau_max)
            tau_range_np = tau_max_np - tau_min_np
            print(f"\n[DEBUG Epoch {epoch}] Torque Statistics (Nm):")
            print(f"  Mean: [{tau_mean_np[0]:.3f}, {tau_mean_np[1]:.3f}, {tau_mean_np[2]:.3f}, {tau_mean_np[3]:.3f}, {tau_mean_np[4]:.3f}]")
            print(f"  Std:  [{tau_std_np[0]:.3f}, {tau_std_np[1]:.3f}, {tau_std_np[2]:.3f}, {tau_std_np[3]:.3f}, {tau_std_np[4]:.3f}]")
            print(f"  Min:  [{tau_min_np[0]:.3f}, {tau_min_np[1]:.3f}, {tau_min_np[2]:.3f}, {tau_min_np[3]:.3f}, {tau_min_np[4]:.3f}]")
            print(f"  Max:  [{tau_max_np[0]:.3f}, {tau_max_np[1]:.3f}, {tau_max_np[2]:.3f}, {tau_max_np[3]:.3f}, {tau_max_np[4]:.3f}]")
            print(f"  Range (Max-Min): [{tau_range_np[0]:.3f}, {tau_range_np[1]:.3f}, {tau_range_np[2]:.3f}, {tau_range_np[3]:.3f}, {tau_range_np[4]:.3f}]")
            # Warning if torque range is too small (potential mode collapse)
            if np.max(tau_range_np[:4]) < 0.1:  # Arm joints should have at least 0.1 Nm range
                print(f"  WARNING: Torque range is very small! Possible mode collapse.")

        # Log to TensorBoard
        writer.add_scalar('Loss/Train_Total', np.array(loss), epoch)
        writer.add_scalar('Loss/Train_Arm', np.array(loss_arm), epoch)
        writer.add_scalar('Loss/Train_Grip', np.array(loss_grip), epoch)
        writer.add_scalar('Loss/Train_Pos', np.array(loss_pos), epoch)
        if vel_loss_weight > 0:
            writer.add_scalar('Loss/Train_Vel', np.array(loss_vel), epoch)
        writer.add_scalar('Metric/Train_MAE_Pos', np.array(mae_pos), epoch)

        # Force/Gate metrics - only log when enabled
        if force_loss_weight > 0 or gate_loss_weight > 0:
            writer.add_scalar('Loss/Train_Force', np.array(loss_force), epoch)
            writer.add_scalar('Loss/Train_Gate', np.array(loss_gate), epoch)
            writer.add_scalar('Metric/Train_MAE_Force', np.array(mae_force), epoch)
            writer.add_scalar('Metric/Train_Gate_Acc', np.array(acc_gate), epoch)
            writer.add_scalar('Metric/Train_HasForceRatio', np.array(has_force_ratio), epoch)

        # Condition loss - only log when enabled
        if condition_loss_weight > 0:
            writer.add_scalar('Loss/Train_Cond', np.array(loss_cond), epoch)

        # Condition classification metrics (for paper reporting) - only log when condition_loss_weight > 0
        if condition_loss_weight > 0:
            writer.add_scalar('Condition/Train_Accuracy', np.array(cond_acc), epoch)
            writer.add_scalar('Condition/Train_Precision', np.array(cond_prec), epoch)
            writer.add_scalar('Condition/Train_Recall', np.array(cond_rec), epoch)
            writer.add_scalar('Condition/Train_F1', np.array(cond_f1), epoch)
            writer.add_scalar('Condition/Train_Specificity', np.array(cond_spec), epoch)
            writer.add_scalar('Condition/Train_BalancedAcc', np.array(cond_bal_acc), epoch)
            writer.add_scalar('Condition/Train_FNR', np.array(cond_fnr), epoch)  # miss rate
            writer.add_scalar('Condition/Train_FPR', np.array(cond_fpr), epoch)  # false alarm rate
            writer.add_scalar('Condition/Train_AUC', np.array(cond_auc), epoch)

        # Log loss weights (fixed)
        writer.add_scalar('Weights/w_arm', np.array(w_arm), epoch)
        writer.add_scalar('Weights/w_grip', np.array(w_grip), epoch)
        writer.add_scalar('Weights/w_force', np.array(w_force), epoch)
        writer.add_scalar('Weights/w_gate', np.array(w_gate), epoch)
        writer.add_scalar('Weights/w_cond', np.array(w_cond), epoch)

        # Log tau statistics for mode collapse detection
        tau_mean_np = np.array(tau_mean)
        tau_std_np = np.array(tau_std)
        tau_range_np = np.array(tau_max) - np.array(tau_min)
        for j in range(5):
            writer.add_scalar(f'Tau/Mean_J{j+1}', tau_mean_np[j], epoch)
            writer.add_scalar(f'Tau/Std_J{j+1}', tau_std_np[j], epoch)
            writer.add_scalar(f'Tau/Range_J{j+1}', tau_range_np[j], epoch)
        # Also log max range across arm joints as a summary metric
        writer.add_scalar('Tau/MaxRange_Arm', np.max(tau_range_np[:4]), epoch)

        # Save Best Train Model
        if loss < min_train_loss:
            min_train_loss = loss
            # Save to log_dir
            best_train_path = os.path.join(log_dir, "best_train_params.pkl")
            with open(best_train_path, 'wb') as f:
                pickle.dump(_checkpoint_payload(state.params), f)
            # Save to outputs/
            best_train_path_out = args.model_out.replace('.pkl', '_best_train.pkl')
            os.makedirs(os.path.dirname(best_train_path_out), exist_ok=True)
            with open(best_train_path_out, 'wb') as f:
                pickle.dump(_checkpoint_payload(state.params), f)
            if epoch % 10 == 0:
                print(f"New best TRAIN model saved (Train Loss: {loss:.4f})")

        # Validation
        if epoch % val_interval == 0 and n_val_samples > 0 and val_boundaries:
            val_start_time = time.time()  # Track validation time separately
            # Sample val batch using boundary-aware sampling (returns start_indices AND traj_starts)
            val_start_indices, val_traj_starts = sample_valid_indices(
                val_boundaries, history_length, rollout_steps, batch_size, rng=np_rng
            )
            val_start_indices_jax = jnp.array(val_start_indices)
            val_batch_traj_starts = jnp.array(val_traj_starts)

            val_batch_gt_pos = []
            val_batch_sensor_data = []
            val_batch_gt_force = []
            val_batch_force_valid = []
            val_batch_cond_gt = []
            val_batch_gt_vel = [] if vel_loss_weight > 0 else None

            for idx in val_start_indices:
                # Target is shifted by 1: we predict torque at t to reach state at t+1
                val_batch_gt_pos.append(val_gt_pos_jax[idx+1:idx+1+rollout_steps])
                val_batch_sensor_data.append(val_data_values_jax[idx:idx+rollout_steps])
                val_batch_gt_force.append(val_gt_force_jax[idx+1:idx+1+rollout_steps])
                val_batch_force_valid.append(val_force_valid_jax[idx+1:idx+1+rollout_steps])
                val_batch_cond_gt.append(val_cond_gt_jax[idx+1:idx+1+rollout_steps])
                if vel_loss_weight > 0:
                    val_batch_gt_vel.append((val_gt_pos_jax[idx+1:idx+1+rollout_steps] - val_gt_pos_jax[idx:idx+rollout_steps]) / data_dt)

            val_batch_gt_pos = jnp.array(val_batch_gt_pos)
            val_batch_sensor_data = jnp.array(val_batch_sensor_data)
            val_batch_gt_force = jnp.array(val_batch_gt_force)
            val_batch_force_valid = jnp.array(val_batch_force_valid)
            val_batch_cond_gt = jnp.array(val_batch_cond_gt)
            if vel_loss_weight > 0:
                val_batch_gt_vel = jnp.array(val_batch_gt_vel)

            # Split RNG for val (though not used for dropout)
            rng, val_rng = jax.random.split(rng)

            val_loss, val_aux = validate_step(state, val_rng, val_start_indices_jax, val_batch_traj_starts, val_batch_gt_pos, val_batch_sensor_data, val_batch_gt_force, val_batch_force_valid, val_batch_cond_gt, val_batch_gt_vel)

            # Unpack val aux (same structure as train, including tau stats and condition metrics)
            val_loss_vel = None
            if vel_loss_weight > 0:
                val_loss_vel = val_aux[-1]
                val_aux = val_aux[:-1]
            (val_loss_arm, val_loss_grip, val_loss_force, val_loss_gate, val_loss_cond, val_mae_pos, val_mae_force, val_per_joint_mae, val_acc_gate,
             val_w_arm, val_w_grip, val_w_force, val_w_gate, val_w_cond, val_has_force_ratio,
             val_tau_mean, val_tau_std, val_tau_min, val_tau_max,
             val_cond_acc, val_cond_prec, val_cond_rec, val_cond_f1,
             val_cond_spec, val_cond_bal_acc, val_cond_fnr, val_cond_fpr, val_cond_auc) = val_aux

            val_loss_pos = val_loss_arm + val_loss_grip

            writer.add_scalar('Loss/Val_Total', np.array(val_loss), epoch)
            writer.add_scalar('Loss/Val_Arm', np.array(val_loss_arm), epoch)
            writer.add_scalar('Loss/Val_Grip', np.array(val_loss_grip), epoch)
            writer.add_scalar('Loss/Val_Pos', np.array(val_loss_pos), epoch)
            if vel_loss_weight > 0:
                writer.add_scalar('Loss/Val_Vel', np.array(val_loss_vel), epoch)
            writer.add_scalar('Metric/Val_MAE_Pos', np.array(val_mae_pos), epoch)

            # Force/Gate metrics - only log when enabled
            if force_loss_weight > 0 or gate_loss_weight > 0:
                writer.add_scalar('Loss/Val_Force', np.array(val_loss_force), epoch)
                writer.add_scalar('Loss/Val_Gate', np.array(val_loss_gate), epoch)
                writer.add_scalar('Metric/Val_MAE_Force', np.array(val_mae_force), epoch)
                writer.add_scalar('Metric/Val_Gate_Acc', np.array(val_acc_gate), epoch)
                writer.add_scalar('Metric/Val_HasForceRatio', np.array(val_has_force_ratio), epoch)

            # Condition loss - only log when enabled
            if condition_loss_weight > 0:
                writer.add_scalar('Loss/Val_Cond', np.array(val_loss_cond), epoch)

            # Condition classification metrics (for paper reporting) - only log when condition_loss_weight > 0
            if condition_loss_weight > 0:
                writer.add_scalar('Condition/Val_Accuracy', np.array(val_cond_acc), epoch)
                writer.add_scalar('Condition/Val_Precision', np.array(val_cond_prec), epoch)
                writer.add_scalar('Condition/Val_Recall', np.array(val_cond_rec), epoch)
                writer.add_scalar('Condition/Val_F1', np.array(val_cond_f1), epoch)
                writer.add_scalar('Condition/Val_Specificity', np.array(val_cond_spec), epoch)
                writer.add_scalar('Condition/Val_BalancedAcc', np.array(val_cond_bal_acc), epoch)
                writer.add_scalar('Condition/Val_FNR', np.array(val_cond_fnr), epoch)  # miss rate
                writer.add_scalar('Condition/Val_FPR', np.array(val_cond_fpr), epoch)  # false alarm rate
                writer.add_scalar('Condition/Val_AUC', np.array(val_cond_auc), epoch)

            # Save Best Validation Model (separate from test-based best model)
            # Note: Test-based best model (_best.pkl) is saved in test evaluation section
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                # Save to log_dir
                best_val_path = os.path.join(log_dir, "best_val_params.pkl")
                with open(best_val_path, 'wb') as f:
                    pickle.dump(_checkpoint_payload(state.params), f)
                # Save to outputs/ with _best_val suffix (not _best, which is reserved for test-based)
                best_val_path_out = args.model_out.replace('.pkl', '_best_val.pkl')
                os.makedirs(os.path.dirname(best_val_path_out), exist_ok=True)
                with open(best_val_path_out, 'wb') as f:
                    pickle.dump(_checkpoint_payload(state.params), f)
                print(f"New best VAL model saved (Val Loss: {val_loss:.4f})")
                print(f"  -> {best_val_path}")
                print(f"  -> {best_val_path_out}")
            
            # Detailed MAE logging
            writer.add_scalar('MAE/pos_val', np.array(val_mae_pos), epoch)
            writer.add_scalar('MAE/force_val', np.array(val_mae_force), epoch)
            
            writer.add_scalar('MAE_Joints/val_j1_deg', np.array(val_per_joint_mae[0]), epoch)
            writer.add_scalar('MAE_Joints/val_j2_deg', np.array(val_per_joint_mae[1]), epoch)
            writer.add_scalar('MAE_Joints/val_j3_deg', np.array(val_per_joint_mae[2]), epoch)
            writer.add_scalar('MAE_Joints/val_j4_deg', np.array(val_per_joint_mae[3]), epoch)
            writer.add_scalar('MAE_Joints/val_grip_mm', np.array(val_per_joint_mae[4]), epoch)
            
            # Accumulate validation time (to exclude from training time)
            total_eval_time += time.time() - val_start_time
        
        if epoch % 10 == 0:
            val_str = ""
            if n_val_samples > 0 and val_loss is not None:
                val_str = f" | Val Loss={val_loss:.4f} (MAE Pos={val_mae_pos:.4f}, Force={val_mae_force:.4f})"

            # Format per-joint MAE for printing
            pj_str = f"J1={per_joint_mae[0]:.2f}deg, J2={per_joint_mae[1]:.2f}deg, J3={per_joint_mae[2]:.2f}deg, J4={per_joint_mae[3]:.2f}deg, Grip={per_joint_mae[4]:.2f}mm"

            # Format detailed loss for printing
            vel_str = f", L_Vel={loss_vel:.4f}" if vel_loss_weight > 0 else ""
            loss_str = f"L_Arm={loss_arm:.4f}, L_Grip={loss_grip:.4f}{vel_str}, L_Force={loss_force:.4f}, L_Gate={loss_gate:.4f}, L_Cond={loss_cond:.4f}"

            # Format learned weights for printing
            weight_str = f"w_arm={w_arm:.2f}, w_grip={w_grip:.2f}, w_force={w_force:.2f}, w_gate={w_gate:.2f}, w_cond={w_cond:.2f}"

            print(f"Epoch {epoch}: Loss={loss:.4f} [{loss_str}] (MAE Pos={mae_pos:.4f}, Force={mae_force:.4f}, GateAcc={acc_gate:.2f}) [{pj_str}]{val_str}")
            print(f"  Learned Weights: [{weight_str}] | HasForceRatio={has_force_ratio:.2%} (Time: {t1-t0:.3f}s)")
            # Print weighted losses for debugging
            weighted_arm = loss_arm * w_arm
            weighted_grip = loss_grip * w_arm * w_grip  # w_arm * w_grip because grip is inside pos_loss
            weighted_force = loss_force * w_force
            weighted_gate = loss_gate * w_gate
            print(f"  Weighted Loss: Arm={weighted_arm:.4f}, Grip={weighted_grip:.4f}, Force={weighted_force:.4f}, Gate={weighted_gate:.4f} | Total={weighted_arm + weighted_grip + weighted_force + weighted_gate:.4f}")

            # Log weighted losses to TensorBoard
            writer.add_scalar('Loss/Weighted_Arm', np.array(weighted_arm), epoch)
            writer.add_scalar('Loss/Weighted_Grip', np.array(weighted_grip), epoch)
            writer.add_scalar('Loss/Weighted_Force', np.array(weighted_force), epoch)
            writer.add_scalar('Loss/Weighted_Gate', np.array(weighted_gate), epoch)

            writer.add_scalar('Loss/train', np.array(loss), epoch)
            writer.add_scalar('Loss/pos_train', np.array(loss_pos), epoch)
            writer.add_scalar('Loss/force_train', np.array(loss_force), epoch)
            writer.add_scalar('Loss/gate_train', np.array(loss_gate), epoch)
            
            writer.add_scalar('MAE/pos_train', np.array(mae_pos), epoch)
            writer.add_scalar('MAE/force_train', np.array(mae_force), epoch)
            
            writer.add_scalar('MAE_Joints/train_j1_deg', np.array(per_joint_mae[0]), epoch)
            writer.add_scalar('MAE_Joints/train_j2_deg', np.array(per_joint_mae[1]), epoch)
            writer.add_scalar('MAE_Joints/train_j3_deg', np.array(per_joint_mae[2]), epoch)
            writer.add_scalar('MAE_Joints/train_j4_deg', np.array(per_joint_mae[3]), epoch)
            writer.add_scalar('MAE_Joints/train_grip_mm', np.array(per_joint_mae[4]), epoch)

        # =====================================================================
        # Save Last Checkpoint (periodically, regardless of performance)
        # =====================================================================
        if save_last_interval > 0 and epoch > 0 and epoch % save_last_interval == 0:
            last_path = os.path.join(log_dir, "last_params.pkl")
            with open(last_path, 'wb') as f:
                pickle.dump(_checkpoint_payload(state.params), f)
            last_path_out = args.model_out.replace('.pkl', '_last.pkl')
            os.makedirs(os.path.dirname(last_path_out), exist_ok=True)
            with open(last_path_out, 'wb') as f:
                pickle.dump(_checkpoint_payload(state.params), f)
            print(f"  [Epoch {epoch}] Last checkpoint saved -> {last_path}")

        # =====================================================================
        # Train Set Evaluation (for Train_Window metrics) - Using MJX for GPU acceleration
        # =====================================================================
        if eval_interval > 0 and train_eval_datasets and epoch > 0 and epoch % eval_interval == 0:
            print(f"\n[Epoch {epoch}] Running train set evaluation (MJX)...")

            eval_params = state.params

            # Load train eval task data
            train_task_data_list = []
            for task_name, csv_path in train_eval_datasets.items():
                if os.path.exists(csv_path):
                    data = load_csv_data(csv_path, float(train_config.get("current_lowpass_alpha", 0.0)))
                    train_task_data_list.append((task_name, data))
                else:
                    print(f"  WARNING: {csv_path} not found, skipping...")

            # Run MJX batch evaluation
            train_results = evaluate_batch_mjx(model, eval_params, train_task_data_list, train_config, mj_model, verbose=False, norm_stats=_NORM_STATS)

            if train_results:
                # Log window-based MAE to TensorBoard
                window_sizes = [10, 100, 200, 300, 400, 500, 600]
                for window in window_sizes:
                    key = f'J1@{window}'
                    if key in list(train_results.values())[0]:
                        avg_w1 = np.mean([r[f'J1@{window}'] for r in train_results.values() if f'J1@{window}' in r])
                        avg_w2 = np.mean([r[f'J2@{window}'] for r in train_results.values() if f'J2@{window}' in r])
                        avg_w3 = np.mean([r[f'J3@{window}'] for r in train_results.values() if f'J3@{window}' in r])
                        avg_w4 = np.mean([r[f'J4@{window}'] for r in train_results.values() if f'J4@{window}' in r])
                        avg_grip = np.mean([r[f'J5@{window}'] for r in train_results.values() if f'J5@{window}' in r])
                        writer.add_scalar(f'Train_Window/J1@{window}', avg_w1, epoch)
                        writer.add_scalar(f'Train_Window/J2@{window}', avg_w2, epoch)
                        writer.add_scalar(f'Train_Window/J3@{window}', avg_w3, epoch)
                        writer.add_scalar(f'Train_Window/J4@{window}', avg_w4, epoch)
                        writer.add_scalar(f'Train_Window/Grip@{window}', avg_grip, epoch)
                        writer.add_scalar(f'Train_Window/Max@{window}', max(avg_w1, avg_w2, avg_w3, avg_w4), epoch)

                # Print summary for full trajectory
                if 'J1' in list(train_results.values())[0]:
                    avg_j1 = np.mean([r['J1'] for r in train_results.values() if 'J1' in r])
                    avg_j2 = np.mean([r['J2'] for r in train_results.values() if 'J2' in r])
                    avg_j3 = np.mean([r['J3'] for r in train_results.values() if 'J3' in r])
                    avg_j4 = np.mean([r['J4'] for r in train_results.values() if 'J4' in r])
                    avg_grip = np.mean([r['J5'] for r in train_results.values() if 'J5' in r])
                    max_err = max(avg_j1, avg_j2, avg_j3, avg_j4)
                    print(f"  Train MAE @Full: J1={avg_j1:.2f}°, J2={avg_j2:.2f}°, J3={avg_j3:.2f}°, J4={avg_j4:.2f}°, Grip={avg_grip:.2f}mm (Max: {max_err:.2f}°)")

        # =====================================================================
        # Test Set Evaluation (for early stopping) - Using MJX for GPU acceleration
        # =====================================================================
        if eval_interval > 0 and test_datasets and epoch > 0 and epoch % eval_interval == 0:
            eval_start_time = time.time()  # Track eval time separately
            print(f"\n[Epoch {epoch}] Running test set evaluation (MJX)...")

            # Use CURRENT model for evaluation (not best model)
            eval_params = state.params

            # Load all task data for batch evaluation
            task_data_list = []
            for task_name, csv_path in test_datasets.items():
                if os.path.exists(csv_path):
                    data = load_csv_data(csv_path, float(train_config.get("current_lowpass_alpha", 0.0)))
                    task_data_list.append((task_name, data))
                else:
                    print(f"  WARNING: {csv_path} not found, skipping...")

            # Run MJX batch evaluation (GPU-accelerated)
            # Pass state.params directly (not wrapped) - evaluate_batch_mjx handles the wrapping
            test_results = evaluate_batch_mjx(model, eval_params, task_data_list, train_config, mj_model, verbose=False, norm_stats=_NORM_STATS)

            # Optionally score the EMA weights on the same tasks (tracked as a
            # separate best-EMA checkpoint; raw selection below is untouched)
            ema_results = None
            if eval_ema_params:
                ema_results = evaluate_batch_mjx(model, _EMA_PARAMS, task_data_list, train_config, mj_model, verbose=False, norm_stats=_NORM_STATS)

            if test_results:
                # Compute average across all tasks using FULL trajectory (for early stopping).
                # Skip non-task entries (e.g. CLASSIFICATION_* metrics from condition eval).
                task_vals = [r for r in test_results.values() if isinstance(r, dict) and 'J1' in r]
                avg_j1 = np.mean([r['J1'] for r in task_vals])
                avg_j2 = np.mean([r['J2'] for r in task_vals])
                avg_j3 = np.mean([r['J3'] for r in task_vals])
                avg_j4 = np.mean([r['J4'] for r in task_vals])
                avg_grip = np.mean([r['J5'] for r in task_vals])
                max_joint_error = max(avg_j1, avg_j2, avg_j3, avg_j4)

                # Log full trajectory MAE (used for early stopping)
                print(f"  Test MAE @Full: J1={avg_j1:.2f}°, J2={avg_j2:.2f}°, J3={avg_j3:.2f}°, J4={avg_j4:.2f}°, Grip={avg_grip:.2f}mm (Max: {max_joint_error:.2f}°, Best: {best_test_mae:.2f}°, Target: <{target_mae_threshold}°)")

                writer.add_scalar('Test/J1_deg', avg_j1, epoch)
                writer.add_scalar('Test/J2_deg', avg_j2, epoch)
                writer.add_scalar('Test/J3_deg', avg_j3, epoch)
                writer.add_scalar('Test/J4_deg', avg_j4, epoch)
                writer.add_scalar('Test/Grip_mm', avg_grip, epoch)
                writer.add_scalar('Test/Max_deg', max_joint_error, epoch)

                # EMA score on the same tasks (logged next to the raw score)
                ema_max_joint_error = None
                if ema_results:
                    ema_vals = [r for r in ema_results.values() if isinstance(r, dict) and 'J1' in r]
                    ema_max_joint_error = max(
                        np.mean([r['J1'] for r in ema_vals]),
                        np.mean([r['J2'] for r in ema_vals]),
                        np.mean([r['J3'] for r in ema_vals]),
                        np.mean([r['J4'] for r in ema_vals]))
                    print(f"  Test @Full: raw Max={max_joint_error:.2f}° | ema Max={ema_max_joint_error:.2f}° (Best EMA: {best_test_ema_mae:.2f}°)")
                    writer.add_scalar('Test/Max_deg_ema', ema_max_joint_error, epoch)

                # Log test condition accuracy (Joint3 only) if available
                if condition_loss_weight > 0 and 'CLASSIFICATION_J3' in test_results:
                    test_cond_acc = test_results['CLASSIFICATION_J3']['accuracy']
                    test_cond_auc = test_results['CLASSIFICATION_J3']['auc_roc']
                    test_cond_prec = test_results['CLASSIFICATION_J3']['precision']
                    test_cond_rec = test_results['CLASSIFICATION_J3']['recall']
                    writer.add_scalar('Condition/Test_Accuracy_J3', test_cond_acc, epoch)
                    writer.add_scalar('Condition/Test_AUC_J3', test_cond_auc, epoch)
                    print(f"  Test Condition (J3): Acc={test_cond_acc*100:.1f}% Prec={test_cond_prec*100:.1f}% Rec={test_cond_rec*100:.1f}% AUC={test_cond_auc:.3f}")

                # Save best model based on configured strategy
                should_save = False
                save_reason = ""

                if save_best_by == 'condition_accuracy':
                    # Save based on condition accuracy (for motor condition task)
                    if condition_loss_weight > 0 and 'CLASSIFICATION_J3' in test_results:
                        current_cond_acc = test_results['CLASSIFICATION_J3']['accuracy']
                        if current_cond_acc > best_test_condition_acc:
                            best_test_condition_acc = current_cond_acc
                            should_save = True
                            save_reason = f"Condition Acc: {current_cond_acc*100:.1f}%"
                    else:
                        # Fallback to MAE if no condition data available
                        if max_joint_error < best_test_mae:
                            best_test_mae = max_joint_error
                            should_save = True
                            save_reason = f"MAE Max: {max_joint_error:.2f}° (no condition data)"
                else:
                    # Default: save based on TEST set MAE (not validation loss)
                    if max_joint_error < best_test_mae:
                        best_test_mae = max_joint_error
                        should_save = True
                        save_reason = f"MAE Max: {max_joint_error:.2f}°"

                if should_save:
                    # Save to log_dir
                    best_test_path = os.path.join(log_dir, "best_test_params.pkl")
                    with open(best_test_path, 'wb') as f:
                        pickle.dump(_checkpoint_payload(state.params), f)
                    # Save to outputs/
                    best_test_path_out = args.model_out.replace('.pkl', '_best_test.pkl')
                    os.makedirs(os.path.dirname(best_test_path_out), exist_ok=True)
                    with open(best_test_path_out, 'wb') as f:
                        pickle.dump(_checkpoint_payload(state.params), f)
                    print(f"  >>> New BEST TEST model saved! ({save_reason})")
                    print(f"      -> {best_test_path}")
                    print(f"      -> {best_test_path_out}")

                # Separate best checkpoint for the EMA weights (selected by test
                # MAE regardless of save_best_by). Same payload as every other
                # checkpoint; the winning weights are in the ema_params field.
                if ema_max_joint_error is not None and ema_max_joint_error < best_test_ema_mae:
                    best_test_ema_mae = ema_max_joint_error
                    best_test_ema_path = os.path.join(log_dir, "best_test_ema_params.pkl")
                    with open(best_test_ema_path, 'wb') as f:
                        pickle.dump(_checkpoint_payload(state.params), f)
                    print(f"  >>> New BEST TEST EMA model saved! (MAE Max: {ema_max_joint_error:.2f}°)")
                    print(f"      -> {best_test_ema_path}")

                # Log window-based MAE to TensorBoard (aggregate)
                window_sizes = [10, 100, 200, 300, 400, 500, 600]
                for window in window_sizes:
                    key = f'J1@{window}'
                    if key in list(test_results.values())[0]:
                        avg_w1 = np.mean([r[f'J1@{window}'] for r in test_results.values() if f'J1@{window}' in r])
                        avg_w2 = np.mean([r[f'J2@{window}'] for r in test_results.values() if f'J2@{window}' in r])
                        avg_w3 = np.mean([r[f'J3@{window}'] for r in test_results.values() if f'J3@{window}' in r])
                        avg_w4 = np.mean([r[f'J4@{window}'] for r in test_results.values() if f'J4@{window}' in r])
                        avg_w_grip = np.mean([r[f'J5@{window}'] for r in test_results.values() if f'J5@{window}' in r])
                        writer.add_scalar(f'Test_Window/J1@{window}', avg_w1, epoch)
                        writer.add_scalar(f'Test_Window/J2@{window}', avg_w2, epoch)
                        writer.add_scalar(f'Test_Window/J3@{window}', avg_w3, epoch)
                        writer.add_scalar(f'Test_Window/J4@{window}', avg_w4, epoch)
                        writer.add_scalar(f'Test_Window/Grip@{window}', avg_w_grip, epoch)
                        writer.add_scalar(f'Test_Window/Max@{window}', max(avg_w1, avg_w2, avg_w3, avg_w4), epoch)

                # Log per-task MAE to TensorBoard and log file (for LaTeX table generation)
                for task_name, r in test_results.items():
                    for window in window_sizes:
                        key = f'J1@{window}'
                        if key in r:
                            writer.add_scalar(f'Test_Task/{task_name}/J1@{window}', r[f'J1@{window}'], epoch)
                            writer.add_scalar(f'Test_Task/{task_name}/J2@{window}', r[f'J2@{window}'], epoch)
                            writer.add_scalar(f'Test_Task/{task_name}/J3@{window}', r[f'J3@{window}'], epoch)
                            writer.add_scalar(f'Test_Task/{task_name}/J4@{window}', r[f'J4@{window}'], epoch)
                            writer.add_scalar(f'Test_Task/{task_name}/Grip@{window}', r[f'J5@{window}'], epoch)
                            # Log force MAE if available (3D force, not just Z)
                            if f'Force@{window}' in r:
                                writer.add_scalar(f'Test_Task/{task_name}/Force@{window}', r[f'Force@{window}'], epoch)

                # Log per-task results to log file (CSV-like format for LaTeX table generation)
                # Check if any task has force data
                any_has_force = any(r.get('has_force', False) for r in test_results.values() if isinstance(r, dict))

                print(f"\n  [Per-Task Results @Full trajectory]")
                if any_has_force:
                    print(f"  {'Task':<25} | {'J1':>6} | {'J2':>6} | {'J3':>6} | {'J4':>6} | {'Grip':>8} | {'Force':>8} | {'Max':>6}")
                    print(f"  {'-'*25}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*6}")
                else:
                    print(f"  {'Task':<25} | {'J1':>6} | {'J2':>6} | {'J3':>6} | {'J4':>6} | {'Grip':>8} | {'Max':>6}")
                    print(f"  {'-'*25}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*8}-+-{'-'*6}")
                for task_name, r in sorted(test_results.items()):
                    if 'J1' in r:
                        t_j1 = r['J1']
                        t_j2 = r['J2']
                        t_j3 = r['J3']
                        t_j4 = r['J4']
                        t_grip = r['J5']
                        t_max = max(t_j1, t_j2, t_j3, t_j4)
                        joint_ok = t_max < target_mae_threshold
                        grip_ok = t_grip < target_gripper_threshold
                        status = "PASS" if (joint_ok and grip_ok) else "FAIL"
                        if any_has_force:
                            t_force = r.get('Force', 0.0)
                            # Show axis-specific force for pushing_gauge tasks
                            force_axis = r.get('ForceAxis', 'ALL')
                            force_label = f"F{force_axis}"
                            print(f"  {task_name:<25} | {t_j1:>5.1f}° | {t_j2:>5.1f}° | {t_j3:>5.1f}° | {t_j4:>5.1f}° | {t_grip:>6.1f}mm | {t_force:>5.2f}N({force_label}) | {t_max:>5.1f}° {status}")
                        else:
                            print(f"  {task_name:<25} | {t_j1:>5.1f}° | {t_j2:>5.1f}° | {t_j3:>5.1f}° | {t_j4:>5.1f}° | {t_grip:>6.1f}mm | {t_max:>5.1f}° {status}")

                # Check early stopping condition (based on FULL trajectory MAE)
                # Check EVERY task's EVERY joint (not average), INCLUDING gripper AND force
                # AND check condition accuracy when target_condition_threshold > 0
                all_tasks_pass = True
                failed_tasks = []
                tasks_evaluated = 0
                for task_name, r in test_results.items():
                    if 'J1' in r:
                        tasks_evaluated += 1
                        task_j1 = r['J1']
                        task_j2 = r['J2']
                        task_j3 = r['J3']
                        task_j4 = r['J4']
                        task_grip = r['J5']  # Gripper in mm
                        task_force = r.get('Force', 0.0)  # 3D Force MAE in N
                        task_max = max(task_j1, task_j2, task_j3, task_j4)
                        # Check joints, gripper, and force separately
                        joint_pass = task_max < target_mae_threshold
                        gripper_pass = task_grip < target_gripper_threshold
                        # Force check: only when target_force_threshold > 0 and task has force data
                        force_pass = True
                        if target_force_threshold > 0 and r.get('has_force', False):
                            force_pass = task_force < target_force_threshold
                        if not joint_pass or not gripper_pass or not force_pass:
                            all_tasks_pass = False
                            failed_tasks.append((task_name, 'Full', task_max, task_j1, task_j2, task_j3, task_j4, task_grip, task_force, joint_pass, gripper_pass, force_pass))

                # If no tasks were evaluated, don't pass
                if tasks_evaluated == 0:
                    all_tasks_pass = False
                    print(f"  [Warning] No tasks have full trajectory data, cannot evaluate early stopping criteria")

                # Check condition accuracy using TEST accuracy (CLASSIFICATION_J3)
                # This is the meaningful metric - joint3 only classification on test set
                condition_pass = True
                if target_condition_threshold > 0 and condition_loss_weight > 0:
                    if 'CLASSIFICATION_J3' in test_results:
                        current_cond_acc_val = test_results['CLASSIFICATION_J3']['accuracy']
                        condition_pass = current_cond_acc_val >= target_condition_threshold
                        if not condition_pass:
                            print(f"  [Condition] Test Accuracy (J3) {current_cond_acc_val*100:.1f}% < target {target_condition_threshold*100:.0f}%")
                    else:
                        # No classification metrics available (e.g., single class in test set)
                        current_cond_acc_val = 1.0
                        print(f"  [Condition] No CLASSIFICATION_J3 in test results, skipping condition check")

                if all_tasks_pass and condition_pass:
                    print("\n" + "=" * 60)
                    print("EARLY STOPPING: Target reached!")
                    print("=" * 60)
                    criteria_str = f"joints < {target_mae_threshold}° AND gripper < {target_gripper_threshold}mm"
                    if target_force_threshold > 0:
                        criteria_str += f" AND forceZ < {target_force_threshold}N"
                    print(f"  ALL tasks @Full trajectory meet: {criteria_str} after {epoch} epochs")
                    print(f"  Avg: J1={avg_j1:.2f}°, J2={avg_j2:.2f}°, J3={avg_j3:.2f}°, J4={avg_j4:.2f}°, Grip={avg_grip:.2f}mm")
                    if target_condition_threshold > 0 and condition_loss_weight > 0:
                        print(f"  Condition accuracy (J3 Test): {current_cond_acc_val*100:.1f}% (target: >{target_condition_threshold*100:.0f}%)")
                    break
                
                # Online learning mode: per-task check (mean J1-J4 < 50% of table baseline)
                if online_learning_mode and online_learning_task_baselines:
                    window = online_learning_window
                    key = f'J1@{window}'
                    
                    if key in list(test_results.values())[0]:
                        # Check each task against its baseline
                        tasks_passed = 0
                        tasks_total = 0
                        print(f"  [ONLINE LEARNING] Per-task progress @{window} steps (target: {online_learning_target_reduction*100:.0f}% reduction):")
                        print(f"    {'Task':<25} | {'Baseline':>8} | {'Target':>8} | {'Current':>8} | {'Reduction':>10} | Status")
                        print(f"    {'-'*25}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}-+-------")
                        
                        for task_name, result in test_results.items():
                            # Find matching baseline (task name might have suffix like _test)
                            baseline = None
                            for bl_name, bl_val in online_learning_task_baselines.items():
                                if bl_name in task_name or task_name in bl_name:
                                    baseline = bl_val
                                    break
                            
                            if baseline is not None:
                                # Compute mean of J1-J4 for this task
                                j1 = result.get(f'J1@{window}', 0)
                                j2 = result.get(f'J2@{window}', 0)
                                j3 = result.get(f'J3@{window}', 0)
                                j4 = result.get(f'J4@{window}', 0)
                                current_mean = np.mean([j1, j2, j3, j4])
                                
                                target = baseline * (1 - online_learning_target_reduction)
                                reduction_pct = (baseline - current_mean) / baseline * 100
                                passed = current_mean <= target
                                
                                tasks_total += 1
                                if passed:
                                    tasks_passed += 1
                                
                                status = "PASS" if passed else "FAIL"
                                print(f"    {task_name:<25} | {baseline:>7.2f}° | {target:>7.2f}° | {current_mean:>7.2f}° | {reduction_pct:>9.1f}% | {status}")
                                
                                # Log to TensorBoard: per-task metrics
                                writer.add_scalar(f'OnlineLearning/{task_name}_error', current_mean, epoch)
                                writer.add_scalar(f'OnlineLearning/{task_name}_reduction_pct', reduction_pct, epoch)
                        
                        # Compute average reduction across all tasks
                        avg_reduction = np.mean([((bl - r.get(f'J1@{window}', 0) + r.get(f'J2@{window}', 0) + r.get(f'J3@{window}', 0) + r.get(f'J4@{window}', 0))/4) / bl * 100 
                                                 for task_name, r in test_results.items() 
                                                 for bl_name, bl in online_learning_task_baselines.items() 
                                                 if bl_name in task_name or task_name in bl_name])
                        
                        # Log aggregate metrics to TensorBoard
                        writer.add_scalar('OnlineLearning/tasks_passed', tasks_passed, epoch)
                        writer.add_scalar('OnlineLearning/tasks_total', tasks_total, epoch)
                        writer.add_scalar('OnlineLearning/pass_rate', tasks_passed / max(tasks_total, 1), epoch)
                        
                        # Save to JSON for plotting
                        online_log_path = os.path.join(log_dir, "online_learning_progress.json")
                        try:
                            with open(online_log_path, 'r') as f:
                                online_log = json.load(f)
                        except:
                            online_log = {'epochs': [], 'per_task': {}}
                        
                        online_log['epochs'].append(epoch)
                        for task_name, result in test_results.items():
                            baseline = None
                            for bl_name, bl_val in online_learning_task_baselines.items():
                                if bl_name in task_name or task_name in bl_name:
                                    baseline = bl_val
                                    break
                            if baseline is not None:
                                j1 = result.get(f'J1@{window}', 0)
                                j2 = result.get(f'J2@{window}', 0)
                                j3 = result.get(f'J3@{window}', 0)
                                j4 = result.get(f'J4@{window}', 0)
                                current_mean = float(np.mean([j1, j2, j3, j4]))  # Convert to Python float
                                reduction_pct = float((baseline - current_mean) / baseline * 100)
                                if task_name not in online_log['per_task']:
                                    online_log['per_task'][task_name] = {'baseline': float(baseline), 'target': float(baseline * 0.5), 'errors': [], 'reductions': []}
                                online_log['per_task'][task_name]['errors'].append(current_mean)
                                online_log['per_task'][task_name]['reductions'].append(reduction_pct)
                        
                        # Compute avg reduction for this epoch and save to JSON
                        epoch_avg_reduction = np.mean([online_log['per_task'][t]['reductions'][-1] 
                                                       for t in online_log['per_task'] 
                                                       if online_log['per_task'][t]['reductions']])
                        if 'avg_reductions' not in online_log:
                            online_log['avg_reductions'] = []
                        online_log['avg_reductions'].append(float(epoch_avg_reduction))
                        
                        with open(online_log_path, 'w') as f:
                            json.dump(online_log, f, indent=2)
                        
                        # Compute average reduction across all tasks
                        all_reductions = []
                        for task_name, result in test_results.items():
                            baseline = None
                            for bl_name, bl_val in online_learning_task_baselines.items():
                                if bl_name in task_name or task_name in bl_name:
                                    baseline = bl_val
                                    break
                            if baseline is not None:
                                j1 = result.get(f'J1@{window}', 0)
                                j2 = result.get(f'J2@{window}', 0)
                                j3 = result.get(f'J3@{window}', 0)
                                j4 = result.get(f'J4@{window}', 0)
                                current_mean = float(np.mean([j1, j2, j3, j4]))
                                reduction_pct = (baseline - current_mean) / baseline * 100
                                all_reductions.append(reduction_pct)
                        
                        avg_reduction_pct = np.mean(all_reductions) if all_reductions else 0
                        print(f"  [ONLINE LEARNING] Progress: {tasks_passed}/{tasks_total} tasks passed | Avg Reduction: {avg_reduction_pct:.1f}% (target: {online_learning_target_reduction*100:.0f}%)")
                        
                        # Log to TensorBoard
                        writer.add_scalar('OnlineLearning/avg_reduction_pct', avg_reduction_pct, epoch)
                        
                        # Early stop when AVERAGE reduction >= target
                        if avg_reduction_pct >= online_learning_target_reduction * 100:
                            total_wall_time = time.time() - training_start_time
                            pure_training_time = total_wall_time - total_eval_time
                            avg_epoch_time = pure_training_time / max(epoch, 1)
                            print("\n" + "=" * 60)
                            print(f"ONLINE LEARNING EARLY STOPPING: Avg reduction {avg_reduction_pct:.1f}% >= {online_learning_target_reduction*100:.0f}%!")
                            print("=" * 60)
                            print(f"  Tasks passed:   {tasks_passed}/{tasks_total}")
                            print(f"  Avg reduction:  {avg_reduction_pct:.1f}%")
                            print(f"  Epochs trained: {epoch}")
                            print(f"  Training time:  {pure_training_time:.1f}s ({pure_training_time/60:.2f} min) [eval excluded]")
                            print(f"  Eval time:      {total_eval_time:.1f}s (excluded)")
                            print(f"  Wall time:      {total_wall_time:.1f}s ({total_wall_time/60:.2f} min)")
                            print(f"  Avg epoch time: {avg_epoch_time:.2f}s")
                            break
                
                if not (all_tasks_pass and condition_pass):
                    # Log failed tasks (only when close to target)
                    if max_joint_error < target_mae_threshold * 1.5:
                        fail_criteria = f"joints>{target_mae_threshold}° or grip>{target_gripper_threshold}mm"
                        if target_force_threshold > 0:
                            fail_criteria += f" or forceZ>{target_force_threshold}N"
                        print(f"  [Per-task check] {len(failed_tasks)} tasks still failing ({fail_criteria}):")
                        for t_name, t_window, t_max, t_j1, t_j2, t_j3, t_j4, t_grip, t_force, j_pass, g_pass, f_pass in sorted(failed_tasks, key=lambda x: -x[2])[:3]:
                            fail_reason = []
                            if not j_pass:
                                fail_reason.append(f"joints={t_max:.1f}°")
                            if not g_pass:
                                fail_reason.append(f"grip={t_grip:.1f}mm")
                            if not f_pass:
                                fail_reason.append(f"force={t_force:.3f}N")
                            print(f"    - {t_name} @{t_window}: {', '.join(fail_reason)} (J1={t_j1:.1f}°, J2={t_j2:.1f}°, J3={t_j3:.1f}°, J4={t_j4:.1f}°, Grip={t_grip:.1f}mm, Force={t_force:.3f}N)")
                
                # Accumulate eval time (to exclude from training time)
                total_eval_time += time.time() - eval_start_time

    # Save Last Model (to both log_dir and outputs/)
    print("Saving last model...")
    # Save to outputs/
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    with open(args.model_out, 'wb') as f:
        pickle.dump(_checkpoint_payload(state.params), f)
    print(f"  -> {args.model_out}")
    # Save to log_dir
    last_model_path = os.path.join(log_dir, "last_params.pkl")
    with open(last_model_path, 'wb') as f:
        pickle.dump(_checkpoint_payload(state.params), f)
    print(f"  -> {last_model_path}")

    # =========================================================================
    # Print Model Statistics
    # =========================================================================
    print("\n" + "=" * 60)
    print("MODEL STATISTICS")
    print("=" * 60)

    # Count parameters
    def count_params(params):
        """Count total number of parameters in a pytree."""
        return sum(x.size for x in jax.tree_util.tree_leaves(params))

    total_params = count_params(state.params)
    print(f"Model type: {model_type}")
    print(f"Total parameters: {total_params:,}")
    if total_params > 1e6:
        print(f"               = {total_params/1e6:.2f}M")
    elif total_params > 1e3:
        print(f"               = {total_params/1e3:.2f}K")

    # Measure inference FPS
    print("\nMeasuring inference speed...")

    # Prepare dummy inputs
    dummy_hist = jnp.ones((1, history_length * feature_dim))
    dummy_curr = jnp.ones((1, feature_dim))
    dummy_h = jnp.zeros((1, hidden_dim))
    if model_type == 'lstm':
        dummy_state = ((dummy_h, dummy_h), (dummy_h, dummy_h))
    elif model_type in ['gru', 'lnn']:
        dummy_state = (dummy_h, dummy_h)
    else:
        dummy_state = None

    # JIT compile inference function
    @jax.jit
    def inference_fn(params, hist, curr, state):
        return model.apply(params, hist, curr, state, ts=data_dt, training=False)

    # Warmup
    for _ in range(10):
        _ = inference_fn(state.params, dummy_hist, dummy_curr, dummy_state)

    # Measure
    n_iters = 1000
    start_time = time.perf_counter()
    for _ in range(n_iters):
        _ = inference_fn(state.params, dummy_hist, dummy_curr, dummy_state)
    # Wait for async execution to complete
    jax.block_until_ready(_)
    end_time = time.perf_counter()

    elapsed = end_time - start_time
    fps = n_iters / elapsed
    latency_ms = (elapsed / n_iters) * 1000

    print(f"Inference FPS: {fps:.1f}")
    print(f"Latency: {latency_ms:.3f} ms/inference")
    print("=" * 60)

    print("\nDone.")

if __name__ == "__main__":
    main()
