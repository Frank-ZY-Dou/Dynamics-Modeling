import argparse
import sys
import glob
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
from evaluate_actuator_so101 import (evaluate_batch_mjx, load_csv_data,
                                     N_JOINTS, VEL_COUNTS_PER_RAD)

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
        cfg: Config dict (optional, for current_source / force options)

    Returns:
        If return_boundaries=False: (data_values, q_traj, v_traj, gt_pos, gt_force, force_valid)
        If return_boundaries=True: (data_values, q_traj, v_traj, gt_pos, gt_force, force_valid, boundaries)
            boundaries: List of (start_idx, end_idx) tuples for each trajectory

    SO-101 version: 42D Feature Vector (6 joints, jaw is joint 6)
    """
    # 42D Feature Vector (SO-101 version):
    # 0-5:   goal_pos1-6 (target joint positions, CONTROL SIGNAL from CSV, rad)
    # 6-11:  pos1-6 (current joint positions, from simulation, rad)
    # 12-17: current1-6 (STS3215 signed load counts by default, from CSV)
    # 18-23: vel1-6 (encoder steps/s; from simulation during rollout)
    # 24-29: volts1-6 (decivolts, from CSV)
    # 30-35: temp1-6 (C, from CSV)
    # 36-41: pos_error1-6 (goal_pos - pos, computed from sim during rollout, rad)
    current_source = cfg.get('current_source', 'load') if cfg else 'load'
    feature_cols = (
        [f'goal_pos{i}' for i in range(1, 7)]
        + [f'pos{i}' for i in range(1, 7)]
        + [f'{current_source}{i}' for i in range(1, 7)]
        + [f'vel{i}' for i in range(1, 7)]
        + [f'volts{i}' for i in range(1, 7)]
        + [f'temp{i}' for i in range(1, 7)]
    )

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

            missing_cols = [c for c in feature_cols if c not in df.columns]
            if missing_cols:
                raise KeyError(f"Missing required columns {missing_cols} in {csv_path}")

            data_values = df[feature_cols].values.astype(np.float64)

            # Compute position error features: goal_pos - pos (indices 0-5 minus 6-11)
            pos_error = data_values[:, 0:6] - data_values[:, 6:12]  # shape: (n, 6)
            data_values = np.concatenate([data_values, pos_error], axis=1)  # 36D -> 42D

            # Optional causal low-pass on the current channels (per trajectory,
            # same filter must be applied at evaluation/deployment time)
            lp_alpha = float(cfg.get('current_lowpass_alpha', 0.0)) if cfg else 0.0
            if lp_alpha > 0:
                cur = data_values[:, 12:18].copy()
                for t in range(1, len(cur)):
                    cur[t] = lp_alpha * cur[t] + (1.0 - lp_alpha) * cur[t - 1]
                data_values[:, 12:18] = cur

            n_samples = len(data_values)

            gt_pos = df[[f'pos{i}' for i in range(1, 7)]].values

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
                # force_valid[i, c] = 1 where channel c is NOT -999 (SO-101 only has a
                # real force_z; x/y stay -999 throughout)
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
                    print(f"  Found {nan_count} NaN values in force data -> filled with 0")
                    force_df = force_df.fillna(0.0)
                gt_force = force_df.values

            q_traj = np.zeros((n_samples, mj_model.nq))
            v_traj = np.zeros((n_samples, mj_model.nv))

            q_traj[:, :N_JOINTS] = gt_pos

            timestamps = df['timestamp'].values
            dt_data = np.mean(np.diff(timestamps))
            v_traj[:-1, :N_JOINTS] = (gt_pos[1:] - gt_pos[:-1]) / dt_data

            data_values_all.append(data_values)
            q_traj_all.append(q_traj)
            v_traj_all.append(v_traj)
            gt_pos_all.append(gt_pos)
            gt_force_all.append(gt_force)
            force_valid_all.append(force_valid)

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
            valid_ranges.append((i, traj_start, min_start, max_start))
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

    return start_indices, traj_starts


def validate_mujoco_joint_limits(mj_model, data_values, margin=0.05):
    """Validate that data joint positions fit within MuJoCo joint limits.

    Positions outside the model limits trigger explosive constraint forces in
    simulation, so training refuses to start on any mismatch (NO FALLBACKS).

    Args:
        mj_model: MuJoCo model (6 hinge joints, radians)
        data_values: (N, 42) feature array; pos1-6 at indices 6-11
        margin: Required margin between data range and model limits (rad)
    """
    print("=" * 60)
    print("[VALIDATION] MuJoCo Joint Limits vs Data Ranges")
    print("=" * 60)
    failures = []
    for j in range(N_JOINTS):
        joint_name = mj_model.joint(j).name
        lo, hi = mj_model.jnt_range[j]
        data_lo = float(data_values[:, 6 + j].min())
        data_hi = float(data_values[:, 6 + j].max())
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


def resolve_task_datasets(train_config):
    """Resolve train/val/test CSV paths from the on-disk split.

    Each entry in config['task_dirs'] contains train/ validation/ test/
    subdirectories with numbered CSVs. The on-disk split is used as-is
    (no ratio re-splitting).
    """
    train_paths = []
    val_paths = []
    test_datasets = {}
    train_eval_datasets = {}
    for task_dir in train_config['task_dirs']:
        task_name = "_".join(os.path.normpath(task_dir).split(os.sep)[-2:])
        task_train = sorted(glob.glob(os.path.join(task_dir, 'train', '*.csv')))
        task_val = sorted(glob.glob(os.path.join(task_dir, 'validation', '*.csv')))
        task_test = sorted(glob.glob(os.path.join(task_dir, 'test', '*.csv')))
        if not task_train:
            raise ValueError(f"No train CSVs found under {task_dir}")
        train_paths.extend(task_train)
        val_paths.extend(task_val)
        if task_test:
            test_datasets[task_name] = task_test[0]
        train_eval_datasets[task_name] = task_train[0]
        print(f"  {task_name}: {len(task_train)} train, {len(task_val)} val, {len(task_test)} test")
    return train_paths, val_paths, test_datasets, train_eval_datasets


def main():
    parser = argparse.ArgumentParser(description='Train Neural Actuator (SO-101) via Diff Sim')
    parser.add_argument('--train_config', type=str, default='configs/so101_weight.yaml', help='Path to training config YAML')
    parser.add_argument('--epochs', type=int, default=None, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--rollout_steps', type=int, default=None, help='Steps per rollout')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--log_dir', type=str, default=None, help='TensorBoard log dir')
    parser.add_argument('--model_out', type=str, default='outputs/neural_actuator_so101_params.pkl', help='Path to save model')
    parser.add_argument('--pretrained_path', type=str, default=None, help='Path to pretrained params (optional)')
    parser.add_argument('--seed', type=int, default=0, help='Random seed (a seed key in the config takes precedence)')
    args = parser.parse_args()

    # 1. Load Config
    print(f"Loading config from {args.train_config}...")
    with open(args.train_config, 'r') as f:
        train_config = yaml.safe_load(f)

    epochs = args.epochs if args.epochs is not None else train_config['epochs']
    batch_size = args.batch_size if args.batch_size is not None else train_config['batch_size']
    rollout_steps = args.rollout_steps if args.rollout_steps is not None else train_config['rollout_steps']
    lr = args.lr if args.lr is not None else float(train_config['lr'])
    # Create log dir with timestamp
    import datetime
    timestamp = datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
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
    # A 'seed' key in the config takes precedence over the --seed flag; default is 0 either way.
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
    # Stability options
    torque_clip = float(train_config.get('torque_clip', 3.0))  # STS3215 scale, per-joint clamp
    qvel_clip = float(train_config.get('qvel_clip', 0.0))  # 0 = disabled
    normalize_features = bool(train_config.get('normalize_features', False))
    mask_invalid_force = bool(train_config.get('mask_invalid_force', False))
    gate_loss_weight = float(train_config['gate_loss_weight'])
    condition_loss_weight = float(train_config.get('condition_loss_weight', 0.0))
    # Model type selection
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
    backbone_activation = train_config['backbone_activation']  # For LNN CfC cell

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

    # Residual Torque Prediction Mode
    # When enabled, network predicts residual: final_torque = base_torque + residual
    # base_torque = (load / 1000) * torque_constant
    # The current-source channel is STS3215 signed load counts (full scale 1000 = 100% duty),
    # so torque_constant is the effective torque in Nm at full-scale load.
    use_residual_torque = bool(train_config.get('use_residual_torque', False))
    torque_constant = float(train_config.get('torque_constant', 0.0))  # Nm at full-scale load for STS3215
    if use_residual_torque:
        print(f"Residual torque mode enabled: final_torque = (load/1000) * {torque_constant} Nm + network_output")
    else:
        print(f"Full torque mode: network directly predicts full torque (clamp +/-{torque_clip} Nm)")

    # Data timestep: actual dt between CSV rows
    base_data_dt = float(train_config['data_dt'])
    # Downsampling factor: take every N-th row from CSV
    downsample_factor = int(train_config['downsample_factor'])
    # Effective data_dt after downsampling
    data_dt = base_data_dt * downsample_factor
    if downsample_factor > 1:
        print(f"Downsampling enabled: factor={downsample_factor}, effective data_dt={data_dt:.4f}s ({1/data_dt:.1f}Hz)")

    # 2. Load MuJoCo Model
    mjcf_path = train_config.get('mjcf_path', 'robot_so101/so101_torque_scene.xml')
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

    # Optionally apply the predicted external force at the grasp point during the rollout,
    # so the payload load enters the dynamics through the force head instead of being
    # absorbed by the torque head. The grasp point is the 'gripperframe' site, offset from
    # the gripper body COM; xfrc_applied acts at the COM, so the wrench is placed at the
    # site by adding the moment (site - COM) x f (see the injection below).
    apply_external_force = bool(train_config.get('apply_external_force', False))
    grasp_site_id = int(mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, 'gripperframe'))
    grasp_body_id = int(mj_model.site_bodyid[grasp_site_id])
    if apply_external_force:
        print(f"External force applied at SO-101 grasp point (site 'gripperframe', body id={grasp_body_id})")

    # 3. Load Data (on-disk train/validation/test split, used as-is)
    if 'task_dirs' in train_config:
        print("Resolving task directories (on-disk train/validation/test split)...")
        csv_paths, val_csv_paths, test_datasets, train_eval_datasets = resolve_task_datasets(train_config)
    else:
        csv_paths = train_config['datasets']
        val_csv_paths = train_config.get('val_datasets', [])
        test_datasets = train_config.get('test_datasets', {})
        train_eval_datasets = train_config.get('train_eval_datasets', {})

    if not val_csv_paths:
        raise ValueError("No validation datasets found; the SO-101 pipeline requires the on-disk validation split.")
    print(f"Train files: {len(csv_paths)}, Val files: {len(val_csv_paths)}, Test tasks: {len(test_datasets)}")

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
    # =========================================================================
    print("\n" + "=" * 60)
    print("LOADING DATA WITH TRAJECTORY BOUNDARIES")
    print("=" * 60)

    result = load_dataset(csv_paths, mj_model, downsample_factor, return_boundaries=True, cfg=train_config)
    if result[0] is None:
        raise ValueError("No valid data loaded from csv_paths.")

    train_data_values, train_q_traj, train_v_traj, train_gt_pos, train_gt_force, train_force_valid, train_boundaries = result
    print(f"Train: {len(train_data_values)} samples, {len(train_boundaries)} trajectories")

    # Per-motor condition labels (1=normal); condition head kept for parity with OMX,
    # no degraded-motor data exists for SO-101 so condition_loss_weight should be 0.
    train_cond_gt = np.ones((len(train_data_values), N_JOINTS), dtype=np.float32)

    print(f"\nLoading validation datasets ({len(val_csv_paths)} files)...")
    val_result = load_dataset(val_csv_paths, mj_model, downsample_factor, return_boundaries=True, cfg=train_config)
    if val_result[0] is None:
        raise ValueError("No valid data loaded from val_csv_paths.")
    val_data_values, val_q_traj, val_v_traj, val_gt_pos, val_gt_force, val_force_valid, val_boundaries = val_result
    val_cond_gt = np.ones((len(val_data_values), N_JOINTS), dtype=np.float32)
    print(f"Val: {len(val_data_values)} samples, {len(val_boundaries)} trajectories")
    print("=" * 60 + "\n")

    # =========================================================================
    # Validate joint limits BEFORE training (NO FALLBACKS)
    # =========================================================================
    validate_mujoco_joint_limits(mj_model, np.concatenate([train_data_values, val_data_values], axis=0))

    # Feature normalization: per-feature z-score with training-set statistics.
    # Applied at the network input only; raw features are kept for the simulator.
    # Stats are saved in checkpoints.
    global _NORM_STATS
    if normalize_features:
        feat_mean = train_data_values.mean(axis=0).astype(np.float32)
        feat_std = train_data_values.std(axis=0).astype(np.float32)
        # Per-channel std floors in native units. Columns that are near-constant in
        # the CSVs are overwritten by the simulator during rollouts with a much
        # wider range; a raw data std would pin those channels to the clip rails
        # and erase the feedback signal.
        std_floor = np.array(
            [0.05] * 6      # goal_pos1-6 (rad)
            + [0.05] * 6    # pos1-6 (rad)
            + [1.0] * 6     # current/load1-6 (counts)
            + [1.0] * 6     # vel1-6 (steps/s)
            + [1.0] * 6     # volts1-6 (decivolts)
            + [1.0] * 6     # temp1-6 (C)
            + [0.05] * 6,   # pos_error1-6 (rad)
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
    feature_dim = train_data_values.shape[1]
    print(f"Total loaded samples: {n_train_samples} (feature_dim={feature_dim})")
    print(f"Train force valid ratio (mean over xyz channels): {train_force_valid.mean():.1%}")

    # Convert Val to JAX
    val_gt_pos_jax = jnp.array(val_gt_pos)
    val_data_values_jax = jnp.array(val_data_values)
    val_gt_force_jax = jnp.array(val_gt_force)
    val_force_valid_jax = jnp.array(val_force_valid)
    val_cond_gt_jax = jnp.array(val_cond_gt)
    n_val_samples = len(val_data_values)
    print(f"Val samples: {n_val_samples}")
    print(f"Val force valid ratio (mean over xyz channels): {val_force_valid.mean():.1%}")

    val_interval = train_config['val_interval']

    # Test set evaluation config (for early stopping)
    eval_interval = train_config['eval_interval']
    save_last_interval = train_config.get('save_last_interval', 100)  # Save last checkpoint every N epochs
    target_mae_threshold = train_config['target_mae_threshold']
    target_force_threshold = train_config.get('target_force_threshold', 0.0)  # Force MAE threshold in N (0=disabled)

    if eval_interval > 0 and test_datasets:
        print(f"Test set evaluation enabled: every {eval_interval} epochs")
        print(f"  Target threshold: all joints < {target_mae_threshold} degrees", end="")
        if target_force_threshold > 0:
            print(f", force < {target_force_threshold} N")
        else:
            print()
        print(f"  Test datasets: {len(test_datasets)} tasks")
        if train_eval_datasets:
            print(f"  Train eval datasets: {len(train_eval_datasets)} tasks")

    # 4. Initialize Model
    print(f"Initializing model: {model_type} (hidden_dim={hidden_dim}, latent_dim={latent_dim}, n_joints={N_JOINTS})")
    model = create_model(
        model_type=model_type,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        dropout_rate=dropout_rate,
        backbone_activation=backbone_activation,
        n_joints=N_JOINTS,
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
    rng, init_rng = jax.random.split(rng)
    dummy_h = jnp.zeros((1, hidden_dim))
    if model_type == 'lstm':
        dummy_state = ((dummy_h, dummy_h), (dummy_h, dummy_h))
    elif model_type in ['gru', 'lnn']:
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

            # New checkpoint format carries normalization stats alongside params.
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
                    for key in loaded_params:
                        if key not in init_params:
                            full_key = f"{prefix}/{key}" if prefix else key
                            print(f"    [SKIP] {full_key} - not in current model")
                    return merged
                else:
                    init_shape = jnp.array(init_params).shape
                    loaded_shape = jnp.array(loaded_params).shape

                    if init_shape == loaded_shape:
                        return loaded_params
                    elif len(init_shape) == len(loaded_shape):
                        init_arr = jnp.array(init_params)
                        loaded_arr = jnp.array(loaded_params)
                        can_expand = all(i >= l for i, l in zip(init_shape, loaded_shape))
                        if can_expand:
                            result = init_arr  # Start with random init
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
        # rng_keys: (batch, steps, 2)
        # batch_traj_starts: (batch,) - trajectory start indices for zero-padding calculation
        # batch_force_valid: (batch, steps, 3) - per channel: 1 where valid, 0 where -999
        # batch_cond_gt: (batch, steps, 6) - motor condition labels (1=normal, 0=degraded)
        # batch_gt_vel: (batch, steps, 6) finite-difference GT velocity, only when vel_loss_weight > 0

        def step_fn(carry, inputs):
            # Unified carry structure: (mjx_data, history_buffer, step_idx, state)
            mjx_data, history_buffer, step_idx, state = carry

            if vel_loss_weight > 0:
                target_pos, target_vel, csv_features, target_force, force_valid_step, cond_gt_step, rng_key = inputs
            else:
                target_pos, csv_features, target_force, force_valid_step, cond_gt_step, rng_key = inputs

            # 1. Construct Current Features (Hybrid)
            q = mjx_data.qpos
            v = mjx_data.qvel

            # 42D Feature Vector (SO-101 version):
            # 0-5:   goal_pos1-6 (from CSV, CONTROL SIGNAL, unchanged)
            # 6-11:  pos1-6 (from simulation)
            # 12-17: current1-6 (from CSV, unchanged)
            # 18-23: vel1-6 (from simulation, converted to encoder steps/s)
            # 24-29: volts1-6 (from CSV, unchanged)
            # 30-35: temp1-6 (from CSV, unchanged)
            # 36-41: pos_error1-6 (goal_pos - pos, computed from sim)
            current_feat = csv_features
            current_feat = current_feat.at[6:12].set(q[:N_JOINTS])
            current_feat = current_feat.at[18:24].set(v[:N_JOINTS] * VEL_COUNTS_PER_RAD)

            # Update position error features from simulation
            pos_error = current_feat[0:6] - q[:N_JOINTS]
            current_feat = current_feat.at[36:42].set(pos_error)

            # 2. Predict Torque & Force (unified interface)
            hist_flat = history_buffer.reshape(-1)

            # RNG for dropout and gumbel
            if training:
                rng_dropout, rng_gumbel = jax.random.split(rng_key)
                rngs = {'dropout': rng_dropout, 'gumbel': rng_gumbel}
            else:
                rngs = None

            # Unified model.apply call
            # Returns 6 values: torque, final_force, raw_force, gate, condition, new_state
            # The history buffer already lives in normalized space; normalize the
            # current frame the same way (identity when normalization is off).
            net_feat = normalize_feat(current_feat)
            tau_pred, final_force, raw_force, gate, condition, new_state = model.apply(
                params, hist_flat[None, :], net_feat[None, :], state,
                ts=data_dt, training=training, rngs=rngs
            )

            # Residual torque mode: final_torque = base_torque + network_output
            # base_torque = (load / 1000) * torque_constant
            # Current-source values are at indices 12-17 in csv_features (load1-6 by default)
            # NOTE: Only apply residual to arm joints (0-4), jaw (5) uses direct prediction
            if use_residual_torque:
                current_values = csv_features[12:17]  # load1-5 in signed counts (arm only)
                # Convert counts to duty fraction, then multiply by torque constant
                base_torque = (current_values / 1000.0) * torque_constant
                # Arm: base_torque + residual, Jaw: direct prediction
                tau = jnp.concatenate([base_torque + tau_pred[0, :5], tau_pred[0, 5:6]])
            else:
                tau = tau_pred[0]  # (6,)

            f_pred = final_force[0] # (3,)
            gate_pred = gate[0, 0] # scalar
            cond_pred = condition[0]  # (6,) - per-motor condition (1=normal, 0=degraded)

            # 3. Step Simulation
            # Clamp torque to prevent simulation divergence from extreme predictions.
            # STS3215 stall torque is ~3 Nm at 12V, use +/-3Nm as the per-joint limit.
            tau_limit = jnp.full(N_JOINTS, torque_clip)
            tau_clamped = jnp.clip(tau, -tau_limit, tau_limit)
            ctrl = jnp.zeros(mjx_model.nu)
            ctrl = ctrl.at[:N_JOINTS].set(tau_clamped)

            mjx_data = mjx_data.replace(ctrl=ctrl)

            # Apply the predicted external force at the grasp point (the 'gripperframe' site).
            # xfrc_applied acts at the gripper body COM, so we add the moment (site - COM) x f
            # to shift the force to the site; MJX carries the per-joint moment from there.
            if apply_external_force:
                r_world = mjx_data.site_xpos[grasp_site_id] - mjx_data.xipos[grasp_body_id]
                xfrc = mjx_data.xfrc_applied.at[grasp_body_id, :3].set(f_pred)
                xfrc = xfrc.at[grasp_body_id, 3:6].set(jnp.cross(r_world, f_pred))
                mjx_data = mjx_data.replace(xfrc_applied=xfrc)

            # Step Simulation (Multi-step)
            def sim_loop_body(i, d):
                return mjx.step(mjx_model, d)

            mjx_data = jax.lax.fori_loop(0, sim_step_size, sim_loop_body, mjx_data)

            # NaN protection: replace any NaN values with target position to prevent gradient corruption
            # This handles simulation divergence gracefully
            qpos_safe = jnp.nan_to_num(mjx_data.qpos, nan=0.0)
            nan_mask = jnp.isnan(mjx_data.qpos[:N_JOINTS])
            qpos_safe = qpos_safe.at[:N_JOINTS].set(jnp.where(nan_mask, target_pos, qpos_safe[:N_JOINTS]))
            mjx_data = mjx_data.replace(qpos=qpos_safe)

            # Optional qvel protection: joint velocities feed back into the network
            # features, so a diverging simulation can blow up training. Clip to a
            # physical bound and scrub NaN/Inf.
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

            # Pose loss over all 6 revolute joints (rad); the jaw is joint 6
            pose_err = jnp.mean(smooth_l1(q_after[:N_JOINTS] - target_pos))

            # Velocity-matching loss (jitter suppression): compare the post-step
            # sim qvel (after the qvel clip above, i.e. exactly the state the
            # rollout carries forward) to the finite-difference GT velocity,
            # over the same 6 joints as the pose loss (rad/s).
            if vel_loss_weight > 0:
                v_after = mjx_data.qvel
                vel_err = jnp.mean(smooth_l1(v_after[:N_JOINTS] - target_vel))

            # Force Loss with Focal Weighting
            # Non-zero force samples get higher weight to combat force imbalance
            # NOTE: -999 samples (converted to 0) are supervised as force=0, gate=0
            force_mag_gt = jnp.sqrt(jnp.sum(target_force**2))
            has_force = (force_mag_gt > 0.01).astype(jnp.float32)
            # Focal weight: force_focal_weight for non-zero, 1.0 for zero
            focal_weight = has_force * (force_focal_weight - 1.0) + 1.0
            # Use Smooth L1 instead of MSE to prevent gradient explosion
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
            gate_gt = has_force
            gate_pred_clipped = jnp.clip(gate_pred, 1e-7, 1.0 - 1e-7)

            # Focal modulation: (1-p_t)^gamma where p_t is predicted prob for true class
            p_t = gate_gt * gate_pred_clipped + (1.0 - gate_gt) * (1.0 - gate_pred_clipped)
            focal_modulation = jnp.power(1.0 - p_t, gate_focal_weight)

            # Class-weighted BCE with focal modulation
            class_weight = gate_gt * gate_pos_weight + (1.0 - gate_gt) * 1.0

            bce = - (gate_gt * jnp.log(gate_pred_clipped) + (1.0 - gate_gt) * jnp.log(1.0 - gate_pred_clipped))
            gate_err = focal_modulation * class_weight * bce

            # Condition Loss (BCE, mean over motors; inert when condition_loss_weight=0)
            cond_gt = cond_gt_step  # (6,)
            cond_pred_clipped = jnp.clip(cond_pred, 1e-7, 1.0 - 1e-7)
            cond_err = jnp.mean(- (cond_gt * jnp.log(cond_pred_clipped) + (1.0 - cond_gt) * jnp.log(1.0 - cond_pred_clipped)))

            pos_mae = jnp.mean(jnp.abs(q_after[:N_JOINTS] - target_pos))
            force_mae = jnp.mean(jnp.abs(f_pred - target_force))

            # Per-joint MAE (degrees)
            diff = jnp.abs(q_after[:N_JOINTS] - target_pos)
            per_joint_mae = diff * 180.0 / jnp.pi

            # Gate Accuracy (for monitoring)
            gate_acc = ((gate_pred > 0.5) == (gate_gt > 0.5)).astype(jnp.float32)

            next_state = new_state if new_state is not None else state
            step_out = (pose_err, force_err, gate_err, cond_err, pos_mae, force_mae, per_joint_mae, gate_acc, has_force, tau)
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
            if init_pos_noise_std > 0:
                perturb_key = rng_seq[0]  # Use first step's key for perturbation
                # Only perturb the arm joints (1-5), not the jaw (joint 6)
                noise = jax.random.normal(perturb_key, shape=(5,)) * init_pos_noise_std
                noise_padded = jnp.concatenate([noise, jnp.zeros(init_q.shape[0] - 5)])
                init_q = init_q + noise_padded

            mjx_data = mjx.make_data(mjx_model)
            mjx_data = mjx_data.replace(qpos=init_q, qvel=init_v)

            # Zero-padding for history buffer when near trajectory start
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
            if model_type == 'lstm':
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
            (pose_losses, force_losses, gate_losses, cond_losses, pos_maes, force_maes, per_joint_maes, gate_accs, has_forces, taus) = scan_out

            # Compute tau statistics for debugging mode collapse
            # taus shape: (rollout_steps, 6)
            tau_mean = jnp.mean(taus, axis=0)
            tau_std = jnp.std(taus, axis=0)
            tau_min = jnp.min(taus, axis=0)
            tau_max = jnp.max(taus, axis=0)

            rollout_out = (jnp.mean(pose_losses), jnp.mean(force_losses), jnp.mean(gate_losses), jnp.mean(cond_losses),
                    jnp.mean(pos_maes), jnp.mean(force_maes), jnp.mean(per_joint_maes, axis=0), jnp.mean(gate_accs),
                    jnp.mean(has_forces), tau_mean, tau_std, tau_min, tau_max)
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
        (batch_pose_loss, batch_force_loss, batch_gate_loss, batch_cond_loss, batch_pos_mae, batch_force_mae,
         batch_per_joint_mae, batch_gate_acc, batch_has_force,
         batch_tau_mean, batch_tau_std, batch_tau_min, batch_tau_max) = vmap_out

        total_pose_loss = jnp.mean(batch_pose_loss)
        total_force_loss = jnp.mean(batch_force_loss)
        total_gate_loss = jnp.mean(batch_gate_loss)
        total_cond_loss = jnp.mean(batch_cond_loss)

        total_pos_mae = jnp.mean(batch_pos_mae)
        total_force_mae = jnp.mean(batch_force_mae)
        total_per_joint_mae = jnp.mean(batch_per_joint_mae, axis=0)
        total_gate_acc = jnp.mean(batch_gate_acc)
        total_has_force_ratio = jnp.mean(batch_has_force)  # Ratio of non-zero force samples

        # Aggregate tau statistics across batch
        total_tau_mean = jnp.mean(batch_tau_mean, axis=0)
        total_tau_std = jnp.mean(batch_tau_std, axis=0)
        total_tau_min = jnp.min(batch_tau_min, axis=0)
        total_tau_max = jnp.max(batch_tau_max, axis=0)

        # Fixed weights loss
        total_loss = (pos_loss_weight * total_pose_loss +
                     force_loss_weight * total_force_loss +
                     gate_loss_weight * total_gate_loss +
                     condition_loss_weight * total_cond_loss)
        if vel_loss_weight > 0:
            total_vel_loss = jnp.mean(batch_vel_loss)
            total_loss = total_loss + vel_loss_weight * total_vel_loss
        w_pos, w_force, w_gate, w_cond = pos_loss_weight, force_loss_weight, gate_loss_weight, condition_loss_weight

        aux = (total_pose_loss, total_force_loss, total_gate_loss, total_cond_loss,
                           total_pos_mae, total_force_mae, total_per_joint_mae, total_gate_acc,
                           w_pos, w_force, w_gate, w_cond, total_has_force_ratio,
                           total_tau_mean, total_tau_std, total_tau_min, total_tau_max)
        if vel_loss_weight > 0:
            aux = aux + (total_vel_loss,)
        return total_loss, aux

    @jax.jit
    def train_step(state, rng, start_idx, batch_traj_starts, batch_gt_pos, batch_sensor_data, batch_gt_force, batch_force_valid, batch_cond_gt, batch_gt_vel=None):
        # Generate RNG keys for dropout: (batch, steps, 2)
        batch_size_local = batch_gt_pos.shape[0]
        steps = batch_gt_pos.shape[1]

        batch_keys = jax.random.split(rng, batch_size_local)
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

    np_rng = np.random.default_rng(seed=42 + seed)

    print("\n[DEBUG] Testing train sampling:")
    _ = sample_valid_indices(train_boundaries, history_length, rollout_steps, batch_size, rng=np_rng, debug=True)

    print("\n[DEBUG] Testing val sampling:")
    _ = sample_valid_indices(val_boundaries, history_length, rollout_steps, batch_size, rng=np_rng, debug=True)

    # Reset RNG for actual training
    np_rng = np.random.default_rng(seed=seed)

    min_val_loss = float('inf')
    min_train_loss = float('inf')
    best_test_mae = float('inf')  # Best test set MAE @Full trajectory (max of J1-J6)
    best_test_ema_mae = float('inf')  # Best test MAE achieved by the EMA weights
    # Initialize validation metrics (for printing when val hasn't run yet)
    val_loss = None
    val_mae_pos = None
    val_mae_force = None

    training_start_time = time.time()
    total_eval_time = 0.0  # Track cumulative eval time (to exclude from training time)

    pbar = tqdm(range(epochs), desc="Training", ncols=140)
    # Optional rollout-length curriculum (paper appendix: 128 -> 256 -> final length).
    # Changing the window length changes batch shapes, so JAX retraces automatically
    # at each stage transition (a few extra compiles over the whole run).
    curriculum_epochs = train_config.get('curriculum_epochs', None)  # e.g. [2000, 5000]
    curriculum_steps = train_config.get('curriculum_steps', None)    # e.g. [128, 256]
    # Alternative: sample the rollout length per epoch from a small fixed set
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
            writer.close()
            sys.exit(1)
        else:
            main._nan_streak = 0

        # Unpack train loss components
        loss_vel = None
        if vel_loss_weight > 0:
            loss_vel = loss_comps[-1]
            loss_comps = loss_comps[:-1]
        (loss_pose, loss_force, loss_gate, loss_cond, mae_pos, mae_force, per_joint_mae, acc_gate,
         w_pos, w_force, w_gate, w_cond, has_force_ratio,
         tau_mean, tau_std, tau_min, tau_max) = loss_comps

        # Update tqdm progress bar
        pbar.set_postfix({
            'loss': f'{loss:.4f}',
            'J1': f'{per_joint_mae[0]:.1f}°',
            'J2': f'{per_joint_mae[1]:.1f}°',
            'J3': f'{per_joint_mae[2]:.1f}°',
            'J4': f'{per_joint_mae[3]:.1f}°',
            'J5': f'{per_joint_mae[4]:.1f}°',
            'J6': f'{per_joint_mae[5]:.1f}°'
        })

        # DEBUG: Print tau statistics every 100 epochs to check for mode collapse
        if epoch % 100 == 0:
            tau_mean_np = np.array(tau_mean)
            tau_std_np = np.array(tau_std)
            tau_min_np = np.array(tau_min)
            tau_max_np = np.array(tau_max)
            tau_range_np = tau_max_np - tau_min_np
            print(f"\n[DEBUG Epoch {epoch}] Torque Statistics (Nm):")
            print(f"  Mean: {np.array2string(tau_mean_np, precision=3)}")
            print(f"  Std:  {np.array2string(tau_std_np, precision=3)}")
            print(f"  Min:  {np.array2string(tau_min_np, precision=3)}")
            print(f"  Max:  {np.array2string(tau_max_np, precision=3)}")
            print(f"  Range (Max-Min): {np.array2string(tau_range_np, precision=3)}")
            if np.max(tau_range_np) < 0.1:
                print(f"  WARNING: Torque range is very small! Possible mode collapse.")

        # Log to TensorBoard
        writer.add_scalar('Loss/Train_Total', np.array(loss), epoch)
        writer.add_scalar('Loss/Train_Pos', np.array(loss_pose), epoch)
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

        # Log loss weights (fixed)
        writer.add_scalar('Weights/w_pos', np.array(w_pos), epoch)
        writer.add_scalar('Weights/w_force', np.array(w_force), epoch)
        writer.add_scalar('Weights/w_gate', np.array(w_gate), epoch)
        writer.add_scalar('Weights/w_cond', np.array(w_cond), epoch)

        # Log tau statistics for mode collapse detection
        tau_mean_np = np.array(tau_mean)
        tau_std_np = np.array(tau_std)
        tau_range_np = np.array(tau_max) - np.array(tau_min)
        for j in range(N_JOINTS):
            writer.add_scalar(f'Tau/Mean_J{j+1}', tau_mean_np[j], epoch)
            writer.add_scalar(f'Tau/Std_J{j+1}', tau_std_np[j], epoch)
            writer.add_scalar(f'Tau/Range_J{j+1}', tau_range_np[j], epoch)
        writer.add_scalar('Tau/MaxRange', np.max(tau_range_np), epoch)

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
            # Sample val batch using boundary-aware sampling
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

            val_loss_vel = None
            if vel_loss_weight > 0:
                val_loss_vel = val_aux[-1]
                val_aux = val_aux[:-1]
            (val_loss_pose, val_loss_force, val_loss_gate, val_loss_cond, val_mae_pos, val_mae_force, val_per_joint_mae, val_acc_gate,
             val_w_pos, val_w_force, val_w_gate, val_w_cond, val_has_force_ratio,
             val_tau_mean, val_tau_std, val_tau_min, val_tau_max) = val_aux

            writer.add_scalar('Loss/Val_Total', np.array(val_loss), epoch)
            writer.add_scalar('Loss/Val_Pos', np.array(val_loss_pose), epoch)
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

            # Save Best Validation Model (separate from test-based best model)
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

            for j in range(N_JOINTS):
                writer.add_scalar(f'MAE_Joints/val_j{j+1}_deg', np.array(val_per_joint_mae[j]), epoch)

            # Accumulate validation time (to exclude from training time)
            total_eval_time += time.time() - val_start_time

        if epoch % 10 == 0:
            val_str = ""
            if n_val_samples > 0 and val_loss is not None:
                val_str = f" | Val Loss={val_loss:.4f} (MAE Pos={val_mae_pos:.4f}, Force={val_mae_force:.4f})"

            # Format per-joint MAE for printing
            pj_str = ", ".join([f"J{j+1}={per_joint_mae[j]:.2f}deg" for j in range(N_JOINTS)])

            # Format detailed loss for printing
            vel_str = f", L_Vel={loss_vel:.4f}" if vel_loss_weight > 0 else ""
            loss_str = f"L_Pos={loss_pose:.4f}{vel_str}, L_Force={loss_force:.4f}, L_Gate={loss_gate:.4f}, L_Cond={loss_cond:.4f}"

            print(f"Epoch {epoch}: Loss={loss:.4f} [{loss_str}] (MAE Pos={mae_pos:.4f}, Force={mae_force:.4f}, GateAcc={acc_gate:.2f}) [{pj_str}]{val_str}")
            # Print weighted losses for debugging
            weighted_pos = loss_pose * w_pos
            weighted_force = loss_force * w_force
            weighted_gate = loss_gate * w_gate
            print(f"  Weighted Loss: Pos={weighted_pos:.4f}, Force={weighted_force:.4f}, Gate={weighted_gate:.4f} | Total={weighted_pos + weighted_force + weighted_gate:.4f} | HasForceRatio={has_force_ratio:.2%} (Time: {t1-t0:.3f}s)")

            # Log weighted losses to TensorBoard
            writer.add_scalar('Loss/Weighted_Pos', np.array(weighted_pos), epoch)
            writer.add_scalar('Loss/Weighted_Force', np.array(weighted_force), epoch)
            writer.add_scalar('Loss/Weighted_Gate', np.array(weighted_gate), epoch)

            writer.add_scalar('MAE/pos_train', np.array(mae_pos), epoch)
            writer.add_scalar('MAE/force_train', np.array(mae_force), epoch)

            for j in range(N_JOINTS):
                writer.add_scalar(f'MAE_Joints/train_j{j+1}_deg', np.array(per_joint_mae[j]), epoch)

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
                    data = load_csv_data(csv_path, train_config.get('current_source', 'load'),
                                         float(train_config.get("current_lowpass_alpha", 0.0)))
                    train_task_data_list.append((task_name, data))
                else:
                    print(f"  WARNING: {csv_path} not found, skipping...")

            # Run MJX batch evaluation
            train_results = evaluate_batch_mjx(model, eval_params, train_task_data_list, train_config, mj_model, verbose=False, norm_stats=_NORM_STATS)

            if train_results:
                # Log window-based MAE to TensorBoard
                window_sizes = [100, 300, 500]
                for window in window_sizes:
                    key = f'J1@{window}'
                    if key in list(train_results.values())[0]:
                        avg_w = [np.mean([r[f'J{j+1}@{window}'] for r in train_results.values() if f'J{j+1}@{window}' in r])
                                 for j in range(N_JOINTS)]
                        for j in range(N_JOINTS):
                            writer.add_scalar(f'Train_Window/J{j+1}@{window}', avg_w[j], epoch)
                        writer.add_scalar(f'Train_Window/Max@{window}', max(avg_w), epoch)

                # Print summary for full trajectory
                if 'J1' in list(train_results.values())[0]:
                    avg_j = [np.mean([r[f'J{j+1}'] for r in train_results.values() if f'J{j+1}' in r])
                             for j in range(N_JOINTS)]
                    avg_str = ", ".join([f"J{j+1}={avg_j[j]:.2f}°" for j in range(N_JOINTS)])
                    print(f"  Train MAE @Full: {avg_str} (Max: {max(avg_j):.2f}°)")

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
                    data = load_csv_data(csv_path, train_config.get('current_source', 'load'),
                                         float(train_config.get("current_lowpass_alpha", 0.0)))
                    task_data_list.append((task_name, data))
                else:
                    print(f"  WARNING: {csv_path} not found, skipping...")

            # Run MJX batch evaluation (GPU-accelerated)
            test_results = evaluate_batch_mjx(model, eval_params, task_data_list, train_config, mj_model, verbose=False, norm_stats=_NORM_STATS)

            # Optionally score the EMA weights on the same tasks (tracked as a
            # separate best-EMA checkpoint; raw selection below is untouched)
            ema_results = None
            if eval_ema_params:
                ema_results = evaluate_batch_mjx(model, _EMA_PARAMS, task_data_list, train_config, mj_model, verbose=False, norm_stats=_NORM_STATS)

            if test_results:
                # Compute average across all tasks using FULL trajectory (for early stopping)
                task_vals = [r for r in test_results.values() if isinstance(r, dict) and 'J1' in r]
                avg_j = [np.mean([r[f'J{j+1}'] for r in task_vals]) for j in range(N_JOINTS)]
                force_vals = [r['Force'] for r in task_vals if r.get('has_force', False)]
                avg_force = np.mean(force_vals) if force_vals else 0.0
                max_joint_error = max(avg_j)

                # Log full trajectory MAE (used for early stopping)
                avg_str = ", ".join([f"J{j+1}={avg_j[j]:.2f}°" for j in range(N_JOINTS)])
                print(f"  Test MAE @Full: {avg_str}, Force={avg_force:.3f}N (Max: {max_joint_error:.2f}°, Best: {best_test_mae:.2f}°, Target: <{target_mae_threshold}°)")

                for j in range(N_JOINTS):
                    writer.add_scalar(f'Test/J{j+1}_deg', avg_j[j], epoch)
                writer.add_scalar('Test/Max_deg', max_joint_error, epoch)
                writer.add_scalar('Test/Force_N', avg_force, epoch)

                # EMA score on the same tasks (logged next to the raw score)
                ema_max_joint_error = None
                if ema_results:
                    ema_vals = [r for r in ema_results.values() if isinstance(r, dict) and 'J1' in r]
                    ema_max_joint_error = max(
                        np.mean([r[f'J{j+1}'] for r in ema_vals]) for j in range(N_JOINTS))
                    print(f"  Test @Full: raw Max={max_joint_error:.2f}° | ema Max={ema_max_joint_error:.2f}° (Best EMA: {best_test_ema_mae:.2f}°)")
                    writer.add_scalar('Test/Max_deg_ema', ema_max_joint_error, epoch)

                # Save best model based on TEST set MAE
                if max_joint_error < best_test_mae:
                    best_test_mae = max_joint_error
                    # Save to log_dir
                    best_test_path = os.path.join(log_dir, "best_test_params.pkl")
                    with open(best_test_path, 'wb') as f:
                        pickle.dump(_checkpoint_payload(state.params), f)
                    # Save to outputs/
                    best_test_path_out = args.model_out.replace('.pkl', '_best_test.pkl')
                    os.makedirs(os.path.dirname(best_test_path_out), exist_ok=True)
                    with open(best_test_path_out, 'wb') as f:
                        pickle.dump(_checkpoint_payload(state.params), f)
                    print(f"  >>> New BEST TEST model saved! (MAE Max: {max_joint_error:.2f}°)")
                    print(f"      -> {best_test_path}")
                    print(f"      -> {best_test_path_out}")

                # Separate best checkpoint for the EMA weights. Same payload as
                # every other checkpoint; the winning weights are in the
                # ema_params field.
                if ema_max_joint_error is not None and ema_max_joint_error < best_test_ema_mae:
                    best_test_ema_mae = ema_max_joint_error
                    best_test_ema_path = os.path.join(log_dir, "best_test_ema_params.pkl")
                    with open(best_test_ema_path, 'wb') as f:
                        pickle.dump(_checkpoint_payload(state.params), f)
                    print(f"  >>> New BEST TEST EMA model saved! (MAE Max: {ema_max_joint_error:.2f}°)")
                    print(f"      -> {best_test_ema_path}")

                # Log window-based MAE to TensorBoard (aggregate)
                window_sizes = [100, 300, 500]
                for window in window_sizes:
                    key = f'J1@{window}'
                    if key in list(test_results.values())[0]:
                        avg_w = [np.mean([r[f'J{j+1}@{window}'] for r in test_results.values() if f'J{j+1}@{window}' in r])
                                 for j in range(N_JOINTS)]
                        for j in range(N_JOINTS):
                            writer.add_scalar(f'Test_Window/J{j+1}@{window}', avg_w[j], epoch)
                        writer.add_scalar(f'Test_Window/Max@{window}', max(avg_w), epoch)

                # Log per-task MAE to TensorBoard and log file
                for task_name, r in test_results.items():
                    for window in window_sizes:
                        key = f'J1@{window}'
                        if key in r:
                            for j in range(N_JOINTS):
                                writer.add_scalar(f'Test_Task/{task_name}/J{j+1}@{window}', r[f'J{j+1}@{window}'], epoch)
                            if f'Force@{window}' in r:
                                writer.add_scalar(f'Test_Task/{task_name}/Force@{window}', r[f'Force@{window}'], epoch)

                # Log per-task results to log file
                any_has_force = any(r.get('has_force', False) for r in test_results.values() if isinstance(r, dict))

                print(f"\n  [Per-Task Results @Full trajectory]")
                header = f"  {'Task':<30} | " + " | ".join([f"{'J' + str(j+1):>6}" for j in range(N_JOINTS)])
                if any_has_force:
                    header += f" | {'Force':>8}"
                header += f" | {'Max':>6}"
                print(header)
                print("  " + "-" * (len(header) - 2))
                for task_name, r in sorted(test_results.items()):
                    if 'J1' in r:
                        t_j = [r[f'J{j+1}'] for j in range(N_JOINTS)]
                        t_max = max(t_j)
                        joint_ok = t_max < target_mae_threshold
                        force_ok = True
                        row = f"  {task_name:<30} | " + " | ".join([f"{v:>5.1f}°" for v in t_j])
                        if any_has_force:
                            t_force = r.get('Force', 0.0)
                            row += f" | {t_force:>6.2f}N"
                            if target_force_threshold > 0 and r.get('has_force', False):
                                force_ok = t_force < target_force_threshold
                        status = "PASS" if (joint_ok and force_ok) else "FAIL"
                        row += f" | {t_max:>5.1f}° {status}"
                        print(row)

                # Check early stopping condition (based on FULL trajectory MAE)
                # Check EVERY task's EVERY joint (not average) AND force when enabled
                all_tasks_pass = True
                failed_tasks = []
                tasks_evaluated = 0
                for task_name, r in test_results.items():
                    if 'J1' in r:
                        tasks_evaluated += 1
                        task_j = [r[f'J{j+1}'] for j in range(N_JOINTS)]
                        task_force = r.get('Force', 0.0)
                        task_max = max(task_j)
                        joint_pass = task_max < target_mae_threshold
                        # Force check: only when target_force_threshold > 0 and task has force data
                        force_pass = True
                        if target_force_threshold > 0 and r.get('has_force', False):
                            force_pass = task_force < target_force_threshold
                        if not joint_pass or not force_pass:
                            all_tasks_pass = False
                            failed_tasks.append((task_name, task_max, task_force, joint_pass, force_pass))

                # If no tasks were evaluated, don't pass
                if tasks_evaluated == 0:
                    all_tasks_pass = False
                    print(f"  [Warning] No tasks have full trajectory data, cannot evaluate early stopping criteria")

                if all_tasks_pass:
                    print("\n" + "=" * 60)
                    print("EARLY STOPPING: Target reached!")
                    print("=" * 60)
                    criteria_str = f"joints < {target_mae_threshold}°"
                    if target_force_threshold > 0:
                        criteria_str += f" AND force < {target_force_threshold}N"
                    print(f"  ALL tasks @Full trajectory meet: {criteria_str} after {epoch} epochs")
                    print(f"  Avg: {avg_str}, Force={avg_force:.3f}N")
                    break

                # Log failed tasks (only when close to target)
                if max_joint_error < target_mae_threshold * 1.5:
                    fail_criteria = f"joints>{target_mae_threshold}°"
                    if target_force_threshold > 0:
                        fail_criteria += f" or force>{target_force_threshold}N"
                    print(f"  [Per-task check] {len(failed_tasks)} tasks still failing ({fail_criteria}):")
                    for t_name, t_max, t_force, j_pass, f_pass in sorted(failed_tasks, key=lambda x: -x[1])[:3]:
                        fail_reason = []
                        if not j_pass:
                            fail_reason.append(f"joints={t_max:.1f}°")
                        if not f_pass:
                            fail_reason.append(f"force={t_force:.3f}N")
                        print(f"    - {t_name}: {', '.join(fail_reason)}")

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
