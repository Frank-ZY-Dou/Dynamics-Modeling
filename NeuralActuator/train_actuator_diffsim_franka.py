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
import yaml
from flax.training import train_state
from torch.utils.tensorboard import SummaryWriter
import pickle
from tqdm import tqdm

from models import create_model, get_model_type_from_config
from evaluate_actuator_franka import (evaluate_batch_mjx, load_csv_data,
                                      N_JOINTS, FINGER_TRAVEL)

# EMA of parameters (set in main() when ema_decay > 0); saved alongside raw params.
_EMA_PARAMS = None


def _checkpoint_payload(params):
    if _EMA_PARAMS is None:
        return params
    return {'params': jax.device_get(params), 'ema_params': jax.device_get(_EMA_PARAMS)}


class TrainState(train_state.TrainState):
    pass

def load_dataset(csv_paths, mj_model, downsample_factor=1, return_boundaries=False, cfg=None):
    """Load dataset from CSV files with optional downsampling.

    Args:
        csv_paths: List of CSV file paths
        mj_model: MuJoCo model
        downsample_factor: Take every N-th row (1 = no downsampling)
        return_boundaries: If True, also return trajectory boundary indices
        cfg: Config dict (for trim / lookahead options)

    Returns:
        If return_boundaries=False: (data_values, q_traj, v_traj, gt_pos, gt_force, force_valid)
        If return_boundaries=True: (data_values, q_traj, v_traj, gt_pos, gt_force, force_valid, boundaries)
            boundaries: List of (start_idx, end_idx) tuples for each trajectory

    Franka version: 52D feature vector (7 arm joints + gripper channels)
    """
    # 52D Feature Vector (Franka version):
    # 0-6:   lookahead_pos1-7 (pos[t+K] * scale, smooth target replacing cmd_pos)
    # 7-13:  pos1-7 (current joint positions, from simulation during rollout)
    # 14:    gripper_width (normalized 0-1)
    # 15-21: tau_d1-7 (commanded torque, Nm)
    # 22-28: vel1-7 (rad/s, from simulation during rollout)
    # 29-35: motor_pos1-7 (motor-side positions)
    # 36-42: motor_vel1-7 (motor-side velocities)
    # 43:    goal_gripper (= gripper_width, no independent goal)
    # 44-50: arm_error1-7 = lookahead_pos - pos
    # 51:    gripper_error = goal_gripper - gripper_width
    #
    # NOTE: cmd_pos in the CSVs is a step function (2-3 setpoints), not a smooth
    # trajectory, so lookahead_pos = pos[t+K] * scale replaces it as the target.
    lookahead_frames = int(cfg.get('lookahead_frames', 5)) if cfg else 5
    lookahead_scale = float(cfg.get('lookahead_scale', 1.03)) if cfg else 1.03

    feature_cols = (
        [f'pos{i}' for i in range(1, 8)]  # placeholder for lookahead (overwritten below)
        + [f'pos{i}' for i in range(1, 8)]
        + ['gripper_width']
        + [f'tau_d{i}' for i in range(1, 8)]
        + [f'vel{i}' for i in range(1, 8)]
        + [f'motor_pos{i}' for i in range(1, 8)]
        + [f'motor_vel{i}' for i in range(1, 8)]
        + ['gripper_width']  # goal_gripper (index 43)
    )

    data_values_all = []
    q_traj_all = []
    v_traj_all = []
    gt_pos_all = []
    gt_force_all = []
    force_valid_all = []

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

            # Trim start/end of trajectory (remove noisy boundaries)
            trim_start = float(cfg.get('trim_start', 0.0)) if cfg else 0.0
            trim_end = float(cfg.get('trim_end', 0.0)) if cfg else 0.0
            if trim_start > 0 or trim_end > 0:
                n = len(df)
                df = df.iloc[int(n * trim_start):int(n * (1 - trim_end))].reset_index(drop=True)
                print(f"  Trimmed: {n} -> {len(df)} rows (start={trim_start:.0%}, end={trim_end:.0%})")

            missing_cols = [c for c in feature_cols if c not in df.columns]
            if missing_cols:
                raise KeyError(f"Missing required columns {sorted(set(missing_cols))} in {csv_path}")

            data_values = df[feature_cols].values.astype(np.float64)

            # Replace indices 0-6 with the lookahead target: pos[t+K] * scale
            pos_data = df[[f'pos{i}' for i in range(1, 8)]].values  # (N, 7)
            n = len(pos_data)
            lookahead_pos = np.zeros_like(pos_data)
            for t in range(n):
                future_t = min(t + lookahead_frames, n - 1)
                lookahead_pos[t] = pos_data[future_t] * lookahead_scale
            data_values[:, 0:7] = lookahead_pos
            print(f"  Lookahead target: pos[t+{lookahead_frames}] * {lookahead_scale}")

            # Error features:
            # arm_error = lookahead_pos - pos (indices 0-6 minus 7-13)
            arm_error = data_values[:, 0:7] - data_values[:, 7:14]  # (n, 7)
            # gripper_error = goal_gripper - gripper_width (index 43 minus 14)
            gripper_error = data_values[:, 43:44] - data_values[:, 14:15]  # (n, 1)
            data_values = np.concatenate([data_values, arm_error, gripper_error], axis=1)  # 44D -> 52D

            n_samples = len(data_values)

            # GT positions: 7 arm joints + gripper finger position in meters
            gt_pos = df[[f'pos{i}' for i in range(1, 8)]].values  # (N, 7)
            gripper_vals = df['gripper_width'].values
            gripper_q = gripper_vals * FINGER_TRAVEL  # normalized [0,1] -> meters [0, 0.04]
            print(f"  Gripper GT: width {gripper_vals.min():.2f}-{gripper_vals.max():.2f} -> q {gripper_q.min():.4f}-{gripper_q.max():.4f}m")
            gt_pos = np.column_stack([gt_pos, gripper_q])  # (N, 8)

            # GT force from the filename weight (e.g. 200_001.csv -> 200 g payload).
            # Lift-and-hold: the external force is the held object's gravity,
            # present for the whole (trimmed) trajectory: [0, 0, -mg].
            csv_basename = os.path.basename(csv_path)
            weight_g = int(csv_basename.split('_')[0])
            force_z = -weight_g / 1000.0 * 9.81
            gt_force = np.zeros((n_samples, 3), dtype=np.float32)
            gt_force[:, 2] = force_z
            force_valid = np.ones(n_samples, dtype=np.float32)
            print(f"  Force GT: {weight_g}g -> [0, 0, {force_z:.3f}] N (all {n_samples} samples valid)")

            q_traj = np.zeros((n_samples, mj_model.nq))  # nq=9 (7 arm + 2 fingers)
            v_traj = np.zeros((n_samples, mj_model.nv))  # nv=9

            q_traj[:, :N_JOINTS] = gt_pos[:, :N_JOINTS]
            # Both finger joints get the same value
            q_traj[:, 7] = gt_pos[:, 7]
            q_traj[:, 8] = gt_pos[:, 7]

            # Velocities from finite differences
            timestamps = df['timestamp'].values
            dt_data = np.mean(np.diff(timestamps))
            v_traj[:-1, :N_JOINTS] = (gt_pos[1:, :N_JOINTS] - gt_pos[:-1, :N_JOINTS]) / dt_data
            v_traj[:-1, 7] = (gt_pos[1:, 7] - gt_pos[:-1, 7]) / dt_data
            v_traj[:-1, 8] = v_traj[:-1, 7]

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

    valid_ratio = force_valid.mean()
    print(f"  Force valid ratio: {valid_ratio:.1%} ({int(force_valid.sum())}/{len(force_valid)} samples)")

    if return_boundaries:
        return data_values, q_traj, v_traj, gt_pos, gt_force, force_valid, boundaries
    return data_values, q_traj, v_traj, gt_pos, gt_force, force_valid


def sample_valid_indices(boundaries, history_length, rollout_steps, batch_size, rng=None, debug=False, uniform_traj=True):
    """Sample start indices that respect trajectory boundaries, with zero-padding support.

    Sampling strategy:
    1. First uniformly random select a trajectory (file)
    2. Then sample start index within that trajectory's valid range
    3. History buffer uses zero-padding for samples near trajectory start

    The rollout window is capped at the trajectory length: trajectories shorter
    than rollout_steps are still sampled and the trailing steps are masked out
    by the caller (per-trajectory valid_steps).

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
        (start_indices, traj_starts, traj_ends):
            start_indices: np.array of start indices
            traj_starts: np.array of trajectory start indices (for zero-padding)
            traj_ends: np.array of trajectory end indices (for valid_steps)
    """
    if rng is None:
        rng = np.random.default_rng()

    if debug:
        print(f"\n[DEBUG sample_valid_indices] (with zero-padding support)")
        print(f"  history_length={history_length}, rollout_steps={rollout_steps}, batch_size={batch_size}")
        print(f"  boundaries ({len(boundaries)} trajectories): {boundaries}")
        print(f"  uniform_traj={uniform_traj}")

    # Compute valid range for each trajectory. With zero-padding, start_idx can
    # be anywhere in the trajectory as long as a (possibly capped) rollout fits.
    valid_ranges = []
    for i, (traj_start, traj_end) in enumerate(boundaries):
        traj_len = traj_end - traj_start
        actual_rollout = min(rollout_steps, traj_len - 1)  # cap at trajectory length
        min_start = traj_start  # Can start from beginning (zero-padded history)
        max_start = traj_end - actual_rollout - 1  # -1 for target offset
        valid_size = max_start - min_start

        if debug:
            print(f"  Traj {i}: [{traj_start}, {traj_end}) len={traj_len}, "
                  f"actual_rollout={actual_rollout}, valid_range=[{min_start}, {max_start}) size={valid_size}")

        if max_start > min_start:
            valid_ranges.append((i, traj_start, min_start, max_start))
        elif traj_len >= 2:
            # Trajectory shorter than rollout: include with start=traj_start (masked)
            valid_ranges.append((i, traj_start, traj_start, traj_start + 1))
            if debug:
                print(f"    -> INCLUDED (shorter than rollout, trailing steps masked)")
        elif debug:
            print(f"    -> SKIPPED (too short, need at least 2 samples)")

    if not valid_ranges:
        raise ValueError(
            f"No valid sampling ranges! Each trajectory must have at least "
            f"2 samples. Boundaries: {boundaries}"
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
    traj_starts = np.zeros(batch_size, dtype=np.int32)
    traj_ends = np.zeros(batch_size, dtype=np.int32)

    for i, vr_idx in enumerate(valid_range_indices):
        traj_idx, traj_start, min_start, max_start = valid_ranges[vr_idx]
        traj_starts[i] = traj_start
        traj_ends[i] = boundaries[traj_idx][1]
        start_indices[i] = rng.integers(min_start, max_start)

    if debug:
        print(f"  Sampled traj_starts: {traj_starts}")
        print(f"  Sampled start_indices: {start_indices}")

    return start_indices, traj_starts, traj_ends


def validate_mujoco_joint_limits(mj_model, data_values, margin=0.05):
    """Validate that data joint positions fit within MuJoCo joint limits.

    Positions outside the model limits trigger explosive constraint forces in
    simulation, so training refuses to start on any mismatch (NO FALLBACKS).

    Args:
        mj_model: MuJoCo model (7 arm hinge joints first, radians)
        data_values: (N, 52) feature array; pos1-7 at indices 7-13
        margin: Required margin between data range and model limits (rad)
    """
    print("=" * 60)
    print("[VALIDATION] MuJoCo Joint Limits vs Data Ranges")
    print("=" * 60)
    failures = []
    for j in range(N_JOINTS):
        joint_name = mj_model.joint(j).name
        lo, hi = mj_model.jnt_range[j]
        data_lo = float(data_values[:, 7 + j].min())
        data_hi = float(data_values[:, 7 + j].max())
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
    parser = argparse.ArgumentParser(description='Train Neural Actuator (Franka Panda) via Diff Sim')
    parser.add_argument('--train_config', type=str, default='configs/franka_lift_hold.yaml', help='Path to training config YAML')
    parser.add_argument('--epochs', type=int, default=None, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--rollout_steps', type=int, default=None, help='Steps per rollout')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--log_dir', type=str, default=None, help='TensorBoard log dir')
    parser.add_argument('--model_out', type=str, default='outputs/neural_actuator_franka_params.pkl', help='Path to save model')
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
    gripper_loss_weight = float(train_config['gripper_loss_weight'])
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
    # differences of the GT positions over the same 7 arm joints as the pose
    # loss. 0 = disabled and leaves the loss graph unchanged.
    vel_loss_weight = float(train_config.get('vel_loss_weight', 0.0))
    if vel_loss_weight > 0:
        print(f"Velocity loss weight: {vel_loss_weight}")
    sim_step_size = int(train_config['sim_step_size'])
    backbone_activation = train_config['backbone_activation']  # For LNN CfC cell

    # Loss clamp: cap total loss to prevent gradient explosion from bad batches
    loss_clamp = float(train_config.get('loss_clamp', 0.0))  # 0 = disabled
    if loss_clamp > 0:
        print(f"Loss clamp enabled: max total_loss = {loss_clamp}")

    # Per-joint pose-loss weights (default: uniform)
    joint_loss_weights_list = train_config.get('joint_loss_weights', None)
    if joint_loss_weights_list is not None:
        joint_loss_weights = jnp.array([float(w) for w in joint_loss_weights_list])
        print(f"Per-joint loss weights: {joint_loss_weights_list}")
    else:
        joint_loss_weights = None
        print("Per-joint loss weights: uniform (1.0 for all)")

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
    # When enabled, network predicts residual: final_torque = tau_d + residual
    # (tau_d is the commanded torque in Nm, feature indices 15-21)
    use_residual_torque = bool(train_config.get('use_residual_torque', False))
    torque_constant = float(train_config.get('torque_constant', 1.0))
    if use_residual_torque:
        print(f"Residual torque mode enabled: final_torque = tau_d * {torque_constant} + network_output")
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

    # 2. Load MuJoCo Model (chdir so the MJCF finds its mesh assets)
    mjcf_path = train_config.get('mjcf_path', 'robot_franka/scene.xml')
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

    # 3. Load Data
    csv_paths = train_config['datasets']
    val_csv_paths = train_config.get('val_datasets', [])
    test_datasets = train_config.get('test_datasets', {})
    train_eval_datasets = train_config.get('train_eval_datasets', {})

    if not val_csv_paths:
        raise ValueError("No validation datasets found; set val_datasets in the config.")
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
                config_hz = 1.0 / base_data_dt
                dt_error = abs(csv_dt_actual - base_data_dt)
                dt_error_pct = dt_error / base_data_dt * 100

                status = "OK" if dt_error < 0.002 else "MISMATCH!"
                print(f"  {os.path.basename(csv_path)}:")
                print(f"    CSV dt:    {csv_dt_actual:.6f}s ({csv_hz:.1f}Hz)")
                print(f"    Config dt: {base_data_dt:.6f}s ({config_hz:.1f}Hz)")
                print(f"    Error:     {dt_error:.6f}s ({dt_error_pct:.1f}%) {status}")

                if dt_error >= 0.002:
                    raise ValueError(
                        f"Data rate mismatch: CSV dt={csv_dt_actual:.4f}s but config data_dt={base_data_dt:.4f}s. "
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
    # no degraded-motor data exists for the Franka so condition_loss_weight should be 0.
    train_cond_gt = np.ones((len(train_data_values), N_JOINTS), dtype=np.float32)

    print(f"\nLoading validation datasets ({len(val_csv_paths)} files)...")
    val_result = load_dataset(val_csv_paths, mj_model, downsample_factor, return_boundaries=True, cfg=train_config)
    if val_result[0] is None:
        raise ValueError("No valid data loaded from val_datasets.")
    val_data_values, val_q_traj, val_v_traj, val_gt_pos, val_gt_force, val_force_valid, val_boundaries = val_result
    val_cond_gt = np.ones((len(val_data_values), N_JOINTS), dtype=np.float32)
    print(f"Val: {len(val_data_values)} samples, {len(val_boundaries)} trajectories")
    print("=" * 60 + "\n")

    # =========================================================================
    # Validate joint limits BEFORE training (NO FALLBACKS)
    # =========================================================================
    validate_mujoco_joint_limits(mj_model, np.concatenate([train_data_values, val_data_values], axis=0))

    # Convert Train to JAX
    gt_pos_jax = jnp.array(train_gt_pos)
    data_values_jax = jnp.array(train_data_values)
    gt_force_jax = jnp.array(train_gt_force)
    cond_gt_jax = jnp.array(train_cond_gt)
    q_traj_jax = jnp.array(train_q_traj)
    v_traj_jax = jnp.array(train_v_traj)

    n_train_samples = len(train_data_values)
    feature_dim = train_data_values.shape[1]
    print(f"Total loaded samples: {n_train_samples} (feature_dim={feature_dim})")

    # Convert Val to JAX
    val_gt_pos_jax = jnp.array(val_gt_pos)
    val_data_values_jax = jnp.array(val_data_values)
    val_gt_force_jax = jnp.array(val_gt_force)
    val_cond_gt_jax = jnp.array(val_cond_gt)
    n_val_samples = len(val_data_values)
    print(f"Val samples: {n_val_samples}")

    val_interval = train_config['val_interval']

    # Test set evaluation config (for early stopping)
    eval_interval = train_config['eval_interval']
    save_last_interval = train_config.get('save_last_interval', 100)  # Save last checkpoint every N epochs
    target_mae_threshold = train_config['target_mae_threshold']
    target_gripper_threshold = train_config.get('target_gripper_threshold', 1.0)  # mm
    target_force_threshold = train_config.get('target_force_threshold', 0.0)  # Force MAE threshold in N (0=disabled)

    if eval_interval > 0 and test_datasets:
        print(f"Test set evaluation enabled: every {eval_interval} epochs")
        print(f"  Target threshold: all joints < {target_mae_threshold} degrees, gripper < {target_gripper_threshold} mm", end="")
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

            # Checkpoints written with EMA tracking store {'params', 'ema_params'}
            if isinstance(loaded_params, dict) and 'ema_params' in loaded_params:
                loaded_params = loaded_params['params']

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

    # Per-joint torque limits from the Panda actuator forcerange
    torque_limits = jnp.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])

    # 5. Define Rollout Loop

    def loss_fn(params, training, rng_keys, start_idx, batch_traj_starts, batch_gt_pos, batch_sensor_data, batch_gt_force, batch_cond_gt, batch_valid_steps, batch_gt_vel=None):
        # rng_keys: (batch, steps, 2)
        # batch_traj_starts: (batch,) - trajectory start indices for zero-padding calculation
        # batch_cond_gt: (batch, steps, 7) - motor condition labels (1=normal, 0=degraded)
        # batch_valid_steps: (batch,) - actual valid rollout steps per sample (rest are padding)
        # batch_gt_vel: (batch, steps, 8) finite-difference GT velocity, only when vel_loss_weight > 0

        def step_fn(carry, inputs):
            # Unified carry structure: (mjx_data, history_buffer, step_idx, state)
            mjx_data, history_buffer, step_idx, state = carry

            if vel_loss_weight > 0:
                target_pos, target_vel, csv_features, target_force, cond_gt_step, rng_key = inputs
            else:
                target_pos, csv_features, target_force, cond_gt_step, rng_key = inputs

            # 1. Construct Current Features (Hybrid)
            q = mjx_data.qpos
            v = mjx_data.qvel

            # 52D Feature Vector (Franka version):
            # 0-6:   lookahead_pos1-7 (from CSV, unchanged during rollout)
            # 7-13:  pos1-7 (from simulation)
            # 14:    gripper_width (from simulation, normalized)
            # 15-21: tau_d1-7 (from CSV, unchanged)
            # 22-28: vel1-7 (from simulation)
            # 29-35: motor_pos1-7 (from CSV, unchanged)
            # 36-42: motor_vel1-7 (from CSV, unchanged)
            # 43:    goal_gripper (from CSV, unchanged)
            # 44-50: arm_error (lookahead_pos - sim pos)
            # 51:    gripper_error (goal_gripper - sim gripper_width)
            current_feat = csv_features
            current_feat = current_feat.at[7:14].set(q[:N_JOINTS])
            current_feat = current_feat.at[14].set(q[7] / FINGER_TRAVEL)
            current_feat = current_feat.at[22:29].set(v[:N_JOINTS])

            arm_error = current_feat[0:7] - q[:N_JOINTS]
            current_feat = current_feat.at[44:51].set(arm_error)
            gripper_error = current_feat[43] - current_feat[14]
            current_feat = current_feat.at[51].set(gripper_error)

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
            tau_pred, final_force, raw_force, gate, condition, new_state = model.apply(
                params, hist_flat[None, :], current_feat[None, :], state,
                ts=data_dt, training=training, rngs=rngs
            )

            # Residual torque mode: final_torque = base_torque + network_output
            # tau_d is at indices 15-21 (Nm already)
            if use_residual_torque:
                tau_d_values = csv_features[15:22]
                base_torque = tau_d_values * torque_constant
                tau = base_torque + tau_pred[0]
            else:
                tau = tau_pred[0]  # (7,)

            f_pred = final_force[0] # (3,)
            gate_pred = gate[0, 0] # scalar
            cond_pred = condition[0]  # (7,) - per-motor condition (1=normal, 0=degraded)

            # 3. Step Simulation
            # Clamp torque per joint to the Panda actuator limits to prevent
            # simulation divergence from extreme predictions
            tau_clamped = jnp.clip(tau, -torque_limits, torque_limits)
            ctrl = jnp.zeros(mjx_model.nu).at[:N_JOINTS].set(tau_clamped)

            mjx_data = mjx_data.replace(ctrl=ctrl)

            # Step Simulation (Multi-step)
            def sim_loop_body(i, d):
                d_new = mjx.step(mjx_model, d)
                # The Panda gripper is a tendon-coupled actuator; with x64
                # enabled MJX promotes its int32 wrap bookkeeping to int64
                # inside step(), which then fails to unify with the carry.
                # Cast back explicitly.
                d_new = d_new.tree_replace({
                    '_impl.ten_wrapadr': d_new._impl.ten_wrapadr.astype(jnp.int32),
                    '_impl.ten_wrapnum': d_new._impl.ten_wrapnum.astype(jnp.int32),
                    '_impl.wrap_obj': d_new._impl.wrap_obj.astype(jnp.int32),
                })
                return d_new

            mjx_data = jax.lax.fori_loop(0, sim_step_size, sim_loop_body, mjx_data)

            # Gripper: directly set to GT position. The Franka parallel gripper
            # is position-controlled on the real robot, so simulation is
            # bypassed and both finger joints track the recorded width. This
            # keeps the gripper from drowning out the arm learning signal.
            gripper_gt = target_pos[7]  # GT finger joint position in meters [0, 0.04]
            clamped_qpos = mjx_data.qpos.at[7].set(gripper_gt)
            clamped_qpos = clamped_qpos.at[8].set(gripper_gt)  # both fingers same

            # NaN protection: replace any NaN values to prevent gradient corruption
            # qpos: use target position as fallback
            qpos_safe = jnp.nan_to_num(clamped_qpos, nan=0.0)
            nan_mask_pos = jnp.isnan(clamped_qpos[:8])
            qpos_safe = qpos_safe.at[:8].set(jnp.where(nan_mask_pos, target_pos, qpos_safe[:8]))
            # qvel: replace NaN/Inf with 0 and clamp to prevent divergence propagation
            qvel_safe = jnp.nan_to_num(mjx_data.qvel, nan=0.0, posinf=0.0, neginf=0.0)
            qvel_safe = jnp.clip(qvel_safe, -100.0, 100.0)
            mjx_data = mjx_data.replace(qpos=qpos_safe, qvel=qvel_safe)

            # 4. Update History
            new_hist = jnp.roll(history_buffer, -1, axis=0)
            new_hist = new_hist.at[-1].set(current_feat)

            # 5. Compute Loss & Metrics
            # IMPORTANT: Use q AFTER stepping, not before!
            q_after = mjx_data.qpos

            # Smooth L1 loss (Huber loss): less sensitive to outliers than MSE
            def smooth_l1(x):
                abs_x = jnp.abs(x)
                return jnp.where(abs_x < 1.0, 0.5 * x**2, abs_x - 0.5)

            # Arm pose loss: 7 arm joints (rad), with optional per-joint weighting
            arm_errs = smooth_l1(q_after[:N_JOINTS] - target_pos[:N_JOINTS])
            if joint_loss_weights is not None:
                arm_err = jnp.sum(arm_errs * joint_loss_weights) / jnp.sum(joint_loss_weights)
            else:
                arm_err = jnp.mean(arm_errs)
            # Gripper error: finger joint vs GT (in mm for comparable magnitude;
            # zero here because the finger qpos is replayed from GT above)
            grip_err = smooth_l1((q_after[7] - target_pos[7]) * 1000.0)

            # Velocity-matching loss: compare the post-step sim qvel (after the
            # qvel clamp above, i.e. exactly the state the rollout carries
            # forward) to the finite-difference GT velocity over the 7 arm
            # joints (rad/s).
            if vel_loss_weight > 0:
                v_after = mjx_data.qvel
                vel_err = jnp.mean(smooth_l1(v_after[:N_JOINTS] - target_vel[:N_JOINTS]))

            # Force Loss: only force_z is supervised (force_x/y carry the -999
            # sentinel in the recordings; the synthesized GT is [0, 0, -mg]).
            # The force term can use its own Huber transition point (beta);
            # at beta = 1.0 this is exactly smooth_l1 above.
            def smooth_l1_force(x):
                if force_huber_beta == 1.0:
                    return smooth_l1(x)
                abs_x = jnp.abs(x)
                return jnp.where(abs_x < force_huber_beta,
                                 0.5 * x**2 / force_huber_beta,
                                 abs_x - 0.5 * force_huber_beta)
            force_z_gt = target_force[2]
            has_force = (jnp.abs(force_z_gt) > 0.01).astype(jnp.float32)
            focal_weight = has_force * (force_focal_weight - 1.0) + 1.0
            force_err = smooth_l1_force(f_pred[2] - force_z_gt) * focal_weight

            # Gate Loss (BCE)
            gate_gt = has_force
            gate_pred_clipped = jnp.clip(gate_pred, 1e-7, 1.0 - 1e-7)
            gate_err = - (gate_gt * jnp.log(gate_pred_clipped) + (1.0 - gate_gt) * jnp.log(1.0 - gate_pred_clipped))

            # Condition Loss (BCE, mean over motors; inert when condition_loss_weight=0)
            cond_gt = cond_gt_step  # (7,)
            cond_pred_clipped = jnp.clip(cond_pred, 1e-7, 1.0 - 1e-7)
            cond_err = jnp.mean(- (cond_gt * jnp.log(cond_pred_clipped) + (1.0 - cond_gt) * jnp.log(1.0 - cond_pred_clipped)))

            # 8D pos comparison: 7 arm + 1 gripper
            sim_pos = jnp.concatenate([q_after[:N_JOINTS], q_after[7:8]])
            pos_mae = jnp.mean(jnp.abs(sim_pos - target_pos))
            force_mae = jnp.abs(f_pred[2] - force_z_gt)  # force_z only

            # Per-joint MAE (7 arm joints in degrees + 1 gripper in mm)
            diff = jnp.abs(sim_pos - target_pos)
            per_joint_mae = jnp.concatenate([diff[:N_JOINTS] * 180.0 / jnp.pi,
                                             diff[7:8] * 1000.0])

            # Gate Accuracy (for monitoring)
            gate_acc = ((gate_pred > 0.5) == (gate_gt > 0.5)).astype(jnp.float32)

            next_state = new_state if new_state is not None else state
            step_out = (arm_err, grip_err, force_err, gate_err, cond_err, pos_mae, force_mae, per_joint_mae, gate_acc, has_force, tau)
            if vel_loss_weight > 0:
                step_out = step_out + (vel_err,)
            return (mjx_data, new_hist, step_idx + 1, next_state), step_out

        # Vmap over batch
        def rollout_single(start_i, traj_start_i, valid_steps_i, gt_pos_seq, sensor_seq, gt_force_seq, cond_gt_seq, rng_seq, gt_vel_seq=None):
            """Rollout single trajectory with zero-padding support for history buffer.

            Args:
                start_i: Start index in global data array
                traj_start_i: Start index of current trajectory (for zero-padding boundary)
                valid_steps_i: Actual number of valid rollout steps (rest are padding)
            """
            init_q = q_traj_jax[start_i]
            init_v = v_traj_jax[start_i]

            # Initial Position Perturbation (Domain Randomization)
            if init_pos_noise_std > 0:
                perturb_key = rng_seq[0]  # Use first step's key for perturbation
                # Only perturb the 7 arm joints, not the fingers
                noise = jax.random.normal(perturb_key, shape=(N_JOINTS,)) * init_pos_noise_std
                noise_padded = jnp.concatenate([noise, jnp.zeros(init_q.shape[0] - N_JOINTS)])
                init_q = init_q + noise_padded

            mjx_data = mjx.make_data(mjx_model)
            mjx_data = mjx_data.replace(qpos=init_q, qvel=init_v)

            # Zero-padding for history buffer when near trajectory start
            hist_start = start_i - jnp.int32(history_length)
            hist_indices = hist_start + jnp.arange(history_length, dtype=jnp.int32)
            valid_mask = hist_indices >= traj_start_i
            safe_indices = jnp.maximum(hist_indices, traj_start_i)
            hist_data = data_values_jax[safe_indices]
            hist_buf = jnp.where(valid_mask[:, None], hist_data, jnp.zeros_like(hist_data))

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
                scan_inputs = (gt_pos_seq, gt_vel_seq, sensor_seq, gt_force_seq, cond_gt_seq, rng_seq)
            else:
                scan_inputs = (gt_pos_seq, sensor_seq, gt_force_seq, cond_gt_seq, rng_seq)
            final_carry, scan_out = jax.lax.scan(
                step_fn,
                init_carry,
                scan_inputs
            )
            if vel_loss_weight > 0:
                vel_losses = scan_out[-1]
                scan_out = scan_out[:-1]
            (arm_losses, grip_losses, force_losses, gate_losses, cond_losses, pos_maes, force_maes, per_joint_maes, gate_accs, has_forces, taus) = scan_out

            # Masked mean: only count valid steps (exclude padding beyond the
            # trajectory end; batches pad short trajectories to rollout_steps)
            n_steps = arm_losses.shape[0]
            step_mask = (jnp.arange(n_steps) < valid_steps_i).astype(jnp.float32)
            mask_sum = jnp.maximum(jnp.sum(step_mask), 1.0)

            def masked_mean(x):
                return jnp.sum(x * step_mask) / mask_sum

            def masked_mean_2d(x):
                return jnp.sum(x * step_mask[:, None], axis=0) / mask_sum

            # Compute tau statistics for debugging mode collapse (valid steps only)
            # taus shape: (rollout_steps, 7)
            valid_taus = jnp.where(step_mask[:, None], taus, 0.0)
            tau_mean = jnp.sum(valid_taus, axis=0) / mask_sum
            tau_std = jnp.sqrt(jnp.sum(step_mask[:, None] * (taus - tau_mean[None, :])**2, axis=0) / mask_sum)
            tau_min = jnp.min(jnp.where(step_mask[:, None], taus, 1e10), axis=0)
            tau_max = jnp.max(jnp.where(step_mask[:, None], taus, -1e10), axis=0)

            rollout_out = (masked_mean(arm_losses), masked_mean(grip_losses), masked_mean(force_losses), masked_mean(gate_losses), masked_mean(cond_losses),
                    masked_mean(pos_maes), masked_mean(force_maes), masked_mean_2d(per_joint_maes), masked_mean(gate_accs),
                    masked_mean(has_forces), tau_mean, tau_std, tau_min, tau_max)
            if vel_loss_weight > 0:
                rollout_out = rollout_out + (masked_mean(vel_losses),)
            return rollout_out

        if vel_loss_weight > 0:
            vmap_out = jax.vmap(rollout_single)(
                start_idx, batch_traj_starts, batch_valid_steps, batch_gt_pos, batch_sensor_data, batch_gt_force, batch_cond_gt, rng_keys, batch_gt_vel
            )
            batch_vel_loss = vmap_out[-1]
            vmap_out = vmap_out[:-1]
        else:
            vmap_out = jax.vmap(rollout_single)(
                start_idx, batch_traj_starts, batch_valid_steps, batch_gt_pos, batch_sensor_data, batch_gt_force, batch_cond_gt, rng_keys
            )
        (batch_arm_loss, batch_grip_loss, batch_force_loss, batch_gate_loss, batch_cond_loss, batch_pos_mae, batch_force_mae,
         batch_per_joint_mae, batch_gate_acc, batch_has_force,
         batch_tau_mean, batch_tau_std, batch_tau_min, batch_tau_max) = vmap_out

        total_arm_loss = jnp.mean(batch_arm_loss)
        total_grip_loss = jnp.mean(batch_grip_loss)
        total_force_loss = jnp.mean(batch_force_loss)
        total_gate_loss = jnp.mean(batch_gate_loss)
        total_cond_loss = jnp.mean(batch_cond_loss)

        total_pos_mae = jnp.mean(batch_pos_mae)
        total_force_mae = jnp.mean(batch_force_mae)
        total_per_joint_mae = jnp.mean(batch_per_joint_mae, axis=0)
        total_gate_acc = jnp.mean(batch_gate_acc)
        total_has_force_ratio = jnp.mean(batch_has_force)

        # Aggregate tau statistics across batch
        total_tau_mean = jnp.mean(batch_tau_mean, axis=0)
        total_tau_std = jnp.mean(batch_tau_std, axis=0)
        total_tau_min = jnp.min(batch_tau_min, axis=0)
        total_tau_max = jnp.max(batch_tau_max, axis=0)

        # Fixed weights loss (gripper term nested inside the pose term)
        total_pose_loss = total_arm_loss + gripper_loss_weight * total_grip_loss
        total_loss = (pos_loss_weight * total_pose_loss +
                     force_loss_weight * total_force_loss +
                     gate_loss_weight * total_gate_loss +
                     condition_loss_weight * total_cond_loss)
        if vel_loss_weight > 0:
            total_vel_loss = jnp.mean(batch_vel_loss)
            total_loss = total_loss + vel_loss_weight * total_vel_loss

        # Loss clamp: cap total loss to prevent gradient explosion from bad batches
        if loss_clamp > 0:
            total_loss = jnp.minimum(total_loss, loss_clamp)

        w_arm, w_grip, w_force, w_gate, w_cond = pos_loss_weight, gripper_loss_weight, force_loss_weight, gate_loss_weight, condition_loss_weight

        aux = (total_arm_loss, total_grip_loss, total_force_loss, total_gate_loss, total_cond_loss,
                           total_pos_mae, total_force_mae, total_per_joint_mae, total_gate_acc,
                           w_arm, w_grip, w_force, w_gate, w_cond, total_has_force_ratio,
                           total_tau_mean, total_tau_std, total_tau_min, total_tau_max)
        if vel_loss_weight > 0:
            aux = aux + (total_vel_loss,)
        return total_loss, aux

    @jax.jit
    def train_step(state, rng, start_idx, batch_traj_starts, batch_gt_pos, batch_sensor_data, batch_gt_force, batch_cond_gt, batch_valid_steps, batch_gt_vel=None):
        # Generate RNG keys for dropout: (batch, steps, 2)
        batch_size_local = batch_gt_pos.shape[0]
        steps = batch_gt_pos.shape[1]

        batch_keys = jax.random.split(rng, batch_size_local)
        rng_keys = jax.vmap(lambda k: jax.random.split(k, steps))(batch_keys)

        def loss_wrapper(params):
            return loss_fn(params, True, rng_keys, start_idx, batch_traj_starts,
                          batch_gt_pos, batch_sensor_data, batch_gt_force, batch_cond_gt, batch_valid_steps,
                          batch_gt_vel)

        (loss, aux), grads = jax.value_and_grad(loss_wrapper, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)

        return state, loss, aux

    @jax.jit
    def validate_step(state, rng, start_idx, batch_traj_starts, batch_gt_pos, batch_sensor_data, batch_gt_force, batch_cond_gt, batch_valid_steps, batch_gt_vel=None):
        batch_size_local = batch_gt_pos.shape[0]
        steps = batch_gt_pos.shape[1]

        batch_keys = jax.random.split(rng, batch_size_local)
        rng_keys = jax.vmap(lambda k: jax.random.split(k, steps))(batch_keys)

        loss, aux = loss_fn(state.params, False, rng_keys, start_idx, batch_traj_starts,
                           batch_gt_pos, batch_sensor_data, batch_gt_force, batch_cond_gt, batch_valid_steps,
                           batch_gt_vel)

        return loss, aux

    def prepare_batch(start_indices, t_ends, gt_pos_arr, sensor_arr, gt_force_arr, cond_gt_arr):
        """Slice per-sample rollout windows, padding short trajectories.

        Trajectories shorter than rollout_steps are padded by repeating the
        last frame; the padded steps are excluded from the loss via
        valid_steps masking. Targets are shifted by 1: torque at t reaches
        the state at t+1.
        """
        batch_gt_pos = []
        batch_sensor_data = []
        batch_gt_force = []
        batch_cond_gt = []
        batch_gt_vel = [] if vel_loss_weight > 0 else None
        valid_steps_list = []

        for idx, t_end in zip(start_indices, t_ends):
            available = t_end - idx - 1  # available rollout steps from this start
            actual = min(rollout_steps, available)
            valid_steps_list.append(actual)

            gt_pos_slice = gt_pos_arr[idx+1:idx+1+actual]
            sensor_slice = sensor_arr[idx:idx+actual]
            gt_force_slice = gt_force_arr[idx+1:idx+1+actual]
            cond_gt_slice = cond_gt_arr[idx+1:idx+1+actual]
            if vel_loss_weight > 0:
                # GT velocity by finite difference, aligned with the shifted
                # targets: vel target at step t is (GT[idx+1+t] - GT[idx+t]) / data_dt
                gt_vel_slice = (gt_pos_arr[idx+1:idx+1+actual] - gt_pos_arr[idx:idx+actual]) / data_dt

            pad_len = rollout_steps - actual
            if pad_len > 0:
                gt_pos_slice = jnp.concatenate([gt_pos_slice, jnp.tile(gt_pos_slice[-1:], (pad_len, 1))])
                sensor_slice = jnp.concatenate([sensor_slice, jnp.tile(sensor_slice[-1:], (pad_len, 1))])
                gt_force_slice = jnp.concatenate([gt_force_slice, jnp.tile(gt_force_slice[-1:], (pad_len, 1))])
                cond_gt_slice = jnp.concatenate([cond_gt_slice, jnp.tile(cond_gt_slice[-1:], (pad_len, 1))])
                if vel_loss_weight > 0:
                    gt_vel_slice = jnp.concatenate([gt_vel_slice, jnp.tile(gt_vel_slice[-1:], (pad_len, 1))])

            batch_gt_pos.append(gt_pos_slice)
            batch_sensor_data.append(sensor_slice)
            batch_gt_force.append(gt_force_slice)
            batch_cond_gt.append(cond_gt_slice)
            if vel_loss_weight > 0:
                batch_gt_vel.append(gt_vel_slice)

        batch_gt_pos = jnp.array(batch_gt_pos)
        batch_sensor_data = jnp.array(batch_sensor_data)
        batch_gt_force = jnp.array(batch_gt_force)
        batch_cond_gt = jnp.array(batch_cond_gt)
        if vel_loss_weight > 0:
            batch_gt_vel = jnp.array(batch_gt_vel)
        batch_valid_steps = jnp.array(valid_steps_list, dtype=jnp.int32)
        return batch_gt_pos, batch_sensor_data, batch_gt_force, batch_cond_gt, batch_valid_steps, batch_gt_vel

    # 6. Training Loop
    writer = SummaryWriter(log_dir)
    print("Starting DiffSim training (Franka Panda)...")

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
    best_test_mae = float('inf')  # Best test set MAE @Full trajectory (max of J1-J7)
    best_test_ema_mae = float('inf')  # Best test MAE achieved by the EMA weights
    # Initialize validation metrics (for printing when val hasn't run yet)
    val_loss = None
    val_mae_pos = None
    val_mae_force = None

    # Trajectory length info (per-traj valid-step masking handles variable lengths)
    min_traj_len = min(end - start for start, end in train_boundaries)
    max_traj_len = max(end - start for start, end in train_boundaries)
    print(f"  Trajectory lengths: min={min_traj_len}, max={max_traj_len} (per-traj masking enabled)")

    pbar = tqdm(range(epochs), desc="Training", ncols=140)
    for epoch in pbar:
        # Sample batch using boundary-aware sampling
        start_indices, traj_starts, traj_ends = sample_valid_indices(
            train_boundaries, history_length, rollout_steps, batch_size,
            rng=np_rng, debug=(epoch == 0)  # Debug output on first epoch only
        )

        (batch_gt_pos, batch_sensor_data, batch_gt_force, batch_cond_gt,
         batch_valid_steps, batch_gt_vel) = prepare_batch(
            start_indices, traj_ends, gt_pos_jax, data_values_jax, gt_force_jax, cond_gt_jax)
        start_indices_jax = jnp.array(start_indices)
        batch_traj_starts = jnp.array(traj_starts)

        t0 = time.time()

        # Split RNG for this step
        rng, step_rng = jax.random.split(rng)

        state, loss, loss_comps = train_step(state, step_rng, start_indices_jax, batch_traj_starts, batch_gt_pos, batch_sensor_data, batch_gt_force, batch_cond_gt, batch_valid_steps, batch_gt_vel)
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
        (loss_arm, loss_grip, loss_force, loss_gate, loss_cond, mae_pos, mae_force, per_joint_mae, acc_gate,
         w_arm, w_grip, w_force, w_gate, w_cond, has_force_ratio,
         tau_mean, tau_std, tau_min, tau_max) = loss_comps

        loss_pos = loss_arm + loss_grip

        # Update tqdm progress bar
        pbar.set_postfix({
            'loss': f'{loss:.4f}',
            'J1': f'{per_joint_mae[0]:.1f}',
            'J2': f'{per_joint_mae[1]:.1f}',
            'J3': f'{per_joint_mae[2]:.1f}',
            'J4': f'{per_joint_mae[3]:.1f}',
            'J5': f'{per_joint_mae[4]:.1f}',
            'J6': f'{per_joint_mae[5]:.1f}',
            'J7': f'{per_joint_mae[6]:.1f}',
            'G': f'{per_joint_mae[7]:.1f}mm'
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
            # Sample val batch using boundary-aware sampling
            val_start_indices, val_traj_starts, val_traj_ends = sample_valid_indices(
                val_boundaries, history_length, rollout_steps, batch_size, rng=np_rng
            )
            val_start_indices_jax = jnp.array(val_start_indices)
            val_batch_traj_starts = jnp.array(val_traj_starts)

            (val_batch_gt_pos, val_batch_sensor_data, val_batch_gt_force, val_batch_cond_gt,
             val_batch_valid_steps, val_batch_gt_vel) = prepare_batch(
                val_start_indices, val_traj_ends, val_gt_pos_jax, val_data_values_jax, val_gt_force_jax, val_cond_gt_jax)

            # Split RNG for val (though not used for dropout)
            rng, val_rng = jax.random.split(rng)

            val_loss, val_aux = validate_step(state, val_rng, val_start_indices_jax, val_batch_traj_starts, val_batch_gt_pos, val_batch_sensor_data, val_batch_gt_force, val_batch_cond_gt, val_batch_valid_steps, val_batch_gt_vel)

            val_loss_vel = None
            if vel_loss_weight > 0:
                val_loss_vel = val_aux[-1]
                val_aux = val_aux[:-1]
            (val_loss_arm, val_loss_grip, val_loss_force, val_loss_gate, val_loss_cond, val_mae_pos, val_mae_force, val_per_joint_mae, val_acc_gate,
             val_w_arm, val_w_grip, val_w_force, val_w_gate, val_w_cond, val_has_force_ratio,
             val_tau_mean, val_tau_std, val_tau_min, val_tau_max) = val_aux

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

            # Save Best Validation Model (separate from test-based best model)
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                # Save to log_dir
                best_val_path = os.path.join(log_dir, "best_val_params.pkl")
                with open(best_val_path, 'wb') as f:
                    pickle.dump(_checkpoint_payload(state.params), f)
                # Save to outputs/ with _best_val suffix
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
            writer.add_scalar('MAE_Joints/val_grip_mm', np.array(val_per_joint_mae[N_JOINTS]), epoch)

        if epoch % 10 == 0:
            val_str = ""
            if n_val_samples > 0 and val_loss is not None:
                val_str = f" | Val Loss={val_loss:.4f} (MAE Pos={val_mae_pos:.4f}, Force={val_mae_force:.4f})"

            # Format per-joint MAE for printing (7 arm joints + gripper)
            pj_parts = [f"J{j+1}={per_joint_mae[j]:.2f}deg" for j in range(N_JOINTS)]
            pj_parts.append(f"Grip={per_joint_mae[N_JOINTS]:.2f}mm")
            pj_str = ", ".join(pj_parts)

            # Format detailed loss for printing
            vel_str = f", L_Vel={loss_vel:.4f}" if vel_loss_weight > 0 else ""
            loss_str = f"L_Arm={loss_arm:.4f}, L_Grip={loss_grip:.4f}{vel_str}, L_Force={loss_force:.4f}, L_Gate={loss_gate:.4f}, L_Cond={loss_cond:.4f}"

            print(f"Epoch {epoch}: Loss={loss:.4f} [{loss_str}] (MAE Pos={mae_pos:.4f}, Force={mae_force:.4f}, GateAcc={acc_gate:.2f}) [{pj_str}]{val_str}")
            # Print weighted losses for debugging
            weighted_arm = loss_arm * w_arm
            weighted_grip = loss_grip * w_arm * w_grip
            weighted_force = loss_force * w_force
            weighted_gate = loss_gate * w_gate
            print(f"  Weighted Loss: Arm={weighted_arm:.4f}, Grip={weighted_grip:.4f}, Force={weighted_force:.4f}, Gate={weighted_gate:.4f} | Total={weighted_arm + weighted_grip + weighted_force + weighted_gate:.4f} | HasForceRatio={has_force_ratio:.2%} (Time: {t1-t0:.3f}s)")

            # Log weighted losses to TensorBoard
            writer.add_scalar('Loss/Weighted_Arm', np.array(weighted_arm), epoch)
            writer.add_scalar('Loss/Weighted_Grip', np.array(weighted_grip), epoch)
            writer.add_scalar('Loss/Weighted_Force', np.array(weighted_force), epoch)
            writer.add_scalar('Loss/Weighted_Gate', np.array(weighted_gate), epoch)

            writer.add_scalar('MAE/pos_train', np.array(mae_pos), epoch)
            writer.add_scalar('MAE/force_train', np.array(mae_force), epoch)

            for j in range(N_JOINTS):
                writer.add_scalar(f'MAE_Joints/train_j{j+1}_deg', np.array(per_joint_mae[j]), epoch)
            writer.add_scalar('MAE_Joints/train_grip_mm', np.array(per_joint_mae[N_JOINTS]), epoch)

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
                    data = load_csv_data(csv_path, train_config)
                    train_task_data_list.append((task_name, data))
                else:
                    print(f"  WARNING: {csv_path} not found, skipping...")

            # Run MJX batch evaluation
            train_results = evaluate_batch_mjx(model, eval_params, train_task_data_list, train_config, mj_model, verbose=False)

            if train_results:
                # Log window-based MAE to TensorBoard
                window_sizes = [10, 100, 200, 300, 400, 500, 600]
                for window in window_sizes:
                    key = f'J1@{window}'
                    if key in list(train_results.values())[0]:
                        avg_w = [np.mean([r[f'J{j+1}@{window}'] for r in train_results.values() if f'J{j+1}@{window}' in r])
                                 for j in range(N_JOINTS)]
                        for j in range(N_JOINTS):
                            writer.add_scalar(f'Train_Window/J{j+1}@{window}', avg_w[j], epoch)
                        avg_w_grip = np.mean([r[f'J8@{window}'] for r in train_results.values() if f'J8@{window}' in r])
                        writer.add_scalar(f'Train_Window/Grip@{window}', avg_w_grip, epoch)
                        writer.add_scalar(f'Train_Window/Max@{window}', max(avg_w), epoch)

                # Print summary for full trajectory
                if 'J1' in list(train_results.values())[0]:
                    avg_j = [np.mean([r[f'J{j+1}'] for r in train_results.values() if f'J{j+1}' in r])
                             for j in range(N_JOINTS)]
                    avg_grip = np.mean([r['J8'] for r in train_results.values() if 'J8' in r])
                    avg_str = ", ".join([f"J{j+1}={avg_j[j]:.2f}" for j in range(N_JOINTS)])
                    print(f"  Train MAE @Full: {avg_str}, Grip={avg_grip:.2f}mm (Max: {max(avg_j):.2f})")

        # =====================================================================
        # Test Set Evaluation (for early stopping) - Using MJX for GPU acceleration
        # =====================================================================
        if eval_interval > 0 and test_datasets and epoch > 0 and epoch % eval_interval == 0:
            print(f"\n[Epoch {epoch}] Running test set evaluation (MJX)...")

            # Use CURRENT model for evaluation (not best model)
            eval_params = state.params

            # Load all task data for batch evaluation
            task_data_list = []
            for task_name, csv_path in test_datasets.items():
                if os.path.exists(csv_path):
                    data = load_csv_data(csv_path, train_config)
                    task_data_list.append((task_name, data))
                else:
                    print(f"  WARNING: {csv_path} not found, skipping...")

            # Run MJX batch evaluation (GPU-accelerated)
            test_results = evaluate_batch_mjx(model, eval_params, task_data_list, train_config, mj_model, verbose=False)

            # Optionally score the EMA weights on the same tasks (tracked as a
            # separate best-EMA checkpoint; raw selection below is untouched)
            ema_results = None
            if eval_ema_params:
                ema_results = evaluate_batch_mjx(model, _EMA_PARAMS, task_data_list, train_config, mj_model, verbose=False)

            if test_results:
                # Compute average across all tasks using FULL trajectory (for early stopping)
                task_vals = [r for r in test_results.values() if isinstance(r, dict) and 'J1' in r]
                avg_j = [np.mean([r[f'J{j+1}'] for r in task_vals]) for j in range(N_JOINTS)]
                avg_grip = np.mean([r['J8'] for r in task_vals])
                force_vals = [r['Force'] for r in task_vals if r.get('has_force', False)]
                avg_force = np.mean(force_vals) if force_vals else 0.0
                max_joint_error = max(avg_j)

                # Log full trajectory MAE (used for early stopping)
                avg_str = ", ".join([f"J{j+1}={avg_j[j]:.2f}" for j in range(N_JOINTS)])
                print(f"  Test MAE @Full: {avg_str}, Grip={avg_grip:.2f}mm, Force={avg_force:.3f}N (Max: {max_joint_error:.2f}, Best: {best_test_mae:.2f}, Target: <{target_mae_threshold})")

                for j in range(N_JOINTS):
                    writer.add_scalar(f'Test/J{j+1}_deg', avg_j[j], epoch)
                writer.add_scalar('Test/Grip_mm', avg_grip, epoch)
                writer.add_scalar('Test/Max_deg', max_joint_error, epoch)
                writer.add_scalar('Test/Force_N', avg_force, epoch)

                # EMA score on the same tasks (logged next to the raw score)
                ema_max_joint_error = None
                if ema_results:
                    ema_vals = [r for r in ema_results.values() if isinstance(r, dict) and 'J1' in r]
                    ema_max_joint_error = max(
                        np.mean([r[f'J{j+1}'] for r in ema_vals]) for j in range(N_JOINTS))
                    print(f"  Test @Full: raw Max={max_joint_error:.2f} | ema Max={ema_max_joint_error:.2f} (Best EMA: {best_test_ema_mae:.2f})")
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
                    print(f"  >>> New BEST TEST model saved! (MAE Max: {max_joint_error:.2f})")
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
                    print(f"  >>> New BEST TEST EMA model saved! (MAE Max: {ema_max_joint_error:.2f})")
                    print(f"      -> {best_test_ema_path}")

                # Log window-based MAE to TensorBoard (aggregate)
                window_sizes = [10, 100, 200, 300, 400, 500, 600]
                for window in window_sizes:
                    key = f'J1@{window}'
                    if key in list(test_results.values())[0]:
                        avg_w = [np.mean([r[f'J{j+1}@{window}'] for r in test_results.values() if f'J{j+1}@{window}' in r])
                                 for j in range(N_JOINTS)]
                        for j in range(N_JOINTS):
                            writer.add_scalar(f'Test_Window/J{j+1}@{window}', avg_w[j], epoch)
                        avg_w_grip = np.mean([r[f'J8@{window}'] for r in test_results.values() if f'J8@{window}' in r])
                        writer.add_scalar(f'Test_Window/Grip@{window}', avg_w_grip, epoch)
                        writer.add_scalar(f'Test_Window/Max@{window}', max(avg_w), epoch)

                # Log per-task MAE to TensorBoard and log file
                for task_name, r in test_results.items():
                    for window in window_sizes:
                        key = f'J1@{window}'
                        if key in r:
                            for j in range(N_JOINTS):
                                writer.add_scalar(f'Test_Task/{task_name}/J{j+1}@{window}', r[f'J{j+1}@{window}'], epoch)
                            writer.add_scalar(f'Test_Task/{task_name}/Grip@{window}', r[f'J8@{window}'], epoch)
                            if f'Force@{window}' in r:
                                writer.add_scalar(f'Test_Task/{task_name}/Force@{window}', r[f'Force@{window}'], epoch)

                # Log per-task results to log file
                any_has_force = any(r.get('has_force', False) for r in test_results.values() if isinstance(r, dict))

                print(f"\n  [Per-Task Results @Full trajectory]")
                header = f"  {'Task':<25} | " + " | ".join([f"{'J' + str(j+1):>6}" for j in range(N_JOINTS)]) + f" | {'Grip':>8}"
                if any_has_force:
                    header += f" | {'Force':>8}"
                header += f" | {'Max':>6}"
                print(header)
                print("  " + "-" * (len(header) - 2))
                for task_name, r in sorted(test_results.items()):
                    if 'J1' in r:
                        t_j = [r[f'J{j+1}'] for j in range(N_JOINTS)]
                        t_grip = r['J8']
                        t_max = max(t_j)
                        joint_ok = t_max < target_mae_threshold
                        grip_ok = t_grip < target_gripper_threshold
                        force_ok = True
                        row = f"  {task_name:<25} | " + " | ".join([f"{v:>5.1f}" for v in t_j]) + f" | {t_grip:>6.1f}mm"
                        if any_has_force:
                            t_force = r.get('Force', 0.0)
                            row += f" | {t_force:>6.2f}N"
                            if target_force_threshold > 0 and r.get('has_force', False):
                                force_ok = t_force < target_force_threshold
                        status = "PASS" if (joint_ok and grip_ok and force_ok) else "FAIL"
                        row += f" | {t_max:>5.1f} {status}"
                        print(row)

                # Check early stopping condition (based on FULL trajectory MAE)
                # Check EVERY task's EVERY joint (not average), gripper, AND force when enabled
                all_tasks_pass = True
                failed_tasks = []
                tasks_evaluated = 0
                for task_name, r in test_results.items():
                    if 'J1' in r:
                        tasks_evaluated += 1
                        task_j = [r[f'J{j+1}'] for j in range(N_JOINTS)]
                        task_grip = r['J8']
                        task_force = r.get('Force', 0.0)
                        task_max = max(task_j)
                        joint_pass = task_max < target_mae_threshold
                        gripper_pass = task_grip < target_gripper_threshold
                        force_pass = True
                        if target_force_threshold > 0 and r.get('has_force', False):
                            force_pass = task_force < target_force_threshold
                        if not joint_pass or not gripper_pass or not force_pass:
                            all_tasks_pass = False
                            failed_tasks.append((task_name, task_max, task_grip, task_force, joint_pass, gripper_pass, force_pass))

                # If no tasks were evaluated, don't pass
                if tasks_evaluated == 0:
                    all_tasks_pass = False
                    print(f"  [Warning] No tasks have full trajectory data, cannot evaluate early stopping criteria")

                if all_tasks_pass:
                    print("\n" + "=" * 60)
                    print("EARLY STOPPING: Target reached!")
                    print("=" * 60)
                    criteria_str = f"joints < {target_mae_threshold} AND gripper < {target_gripper_threshold}mm"
                    if target_force_threshold > 0:
                        criteria_str += f" AND force < {target_force_threshold}N"
                    print(f"  ALL tasks @Full trajectory meet: {criteria_str} after {epoch} epochs")
                    print(f"  Avg: {avg_str}, Grip={avg_grip:.2f}mm, Force={avg_force:.3f}N")
                    break

                # Log failed tasks (only when close to target)
                if max_joint_error < target_mae_threshold * 1.5:
                    fail_criteria = f"joints>{target_mae_threshold} or grip>{target_gripper_threshold}mm"
                    if target_force_threshold > 0:
                        fail_criteria += f" or force>{target_force_threshold}N"
                    print(f"  [Per-task check] {len(failed_tasks)} tasks still failing ({fail_criteria}):")
                    for t_name, t_max, t_grip, t_force, j_pass, g_pass, f_pass in sorted(failed_tasks, key=lambda x: -x[1])[:3]:
                        fail_reason = []
                        if not j_pass:
                            fail_reason.append(f"joints={t_max:.1f}")
                        if not g_pass:
                            fail_reason.append(f"grip={t_grip:.1f}mm")
                        if not f_pass:
                            fail_reason.append(f"force={t_force:.3f}N")
                        print(f"    - {t_name}: {', '.join(fail_reason)}")

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
