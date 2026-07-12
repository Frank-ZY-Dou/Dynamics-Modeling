"""
Evaluate Neural Actuator (SO-101 version) on test sets.

Computes per-joint MAE for each task and outputs results in JSON format.
Joint 1-6: Revolute joints (degrees), joint 6 is the gripper jaw.
"""

import argparse
import glob
import json
import os
import pickle
import hashlib

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np
import pandas as pd
import yaml

# Module-level cache for JIT-compiled evaluation functions
_eval_cache = {
    'mjx_model': None,
    'mjx_data_single': None,
    'eval_fn': None,
    'cache_key': None,
    'max_len': None,
}


def clear_eval_cache():
    """Clear the JIT evaluation cache (useful when model architecture changes)."""
    global _eval_cache
    _eval_cache = {
        'mjx_model': None,
        'mjx_data_single': None,
        'eval_fn': None,
        'cache_key': None,
        'max_len': None,
    }

from models import create_model, get_model_type_from_config

# Constants
N_JOINTS = 6
FEATURE_DIM = 42
# Feetech STS3215 velocity telemetry is in encoder steps/s (4096 steps per rev).
# Simulation qvel (rad/s) is converted to the same unit when writing the vel slots.
VEL_COUNTS_PER_RAD = 4096.0 / (2.0 * np.pi)


def load_model(model_path, config):
    """Load trained model from pickle file.

    Returns (model, params, history_length, feature_dim, norm_stats).
    norm_stats is (mean, std) when the checkpoint was trained with feature
    normalization, else None.
    """
    with open(model_path, 'rb') as f:
        params = pickle.load(f)

    norm_stats = None
    if isinstance(params, dict) and ('feature_mean' in params or 'ema_params' in params):
        if 'feature_mean' in params:
            norm_stats = (np.asarray(params['feature_mean']), np.asarray(params['feature_std']))
        if config.get('use_ema_params', False) and 'ema_params' in params:
            print("  Using EMA parameters from checkpoint")
            params = params['ema_params']
        else:
            params = params['params']

    model_type = get_model_type_from_config(config)
    hidden_dim = config['hidden_dim']
    latent_dim = config['latent_dim']
    history_length = config['history_length']
    dropout_rate = config['dropout_rate']

    # SO-101 version: 42D features (6 joints, no gripper aperture channels)
    feature_dim = config.get('feature_dim', FEATURE_DIM)

    # Create model using factory function
    model = create_model(
        model_type=model_type,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        dropout_rate=dropout_rate,
        n_joints=N_JOINTS,
        # Transformer-specific (use defaults for non-transformer models)
        num_heads=config.get('num_heads', 4),
        num_layers=config.get('num_layers', 2),
        d_ff=config.get('d_ff', 256),
        pool_type=config.get('pool_type', 'mean'),
        use_gated_attention=config.get('use_gated_attention', True),
        # LNN-specific
        backbone_activation=config.get('backbone_activation', 'silu'),
    )

    return model, params, history_length, feature_dim, norm_stats


def load_csv_data(csv_path, current_source='load', current_lowpass_alpha=0.0):
    """Load CSV data and extract relevant columns."""
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    # Joint positions (pos1-6)
    pos_cols = [f'pos{i}' for i in range(1, 7)]
    positions = df[pos_cols].values

    # Goal positions (target)
    goal_cols = [f'goal_pos{i}' for i in range(1, 7)]
    if all(c in df.columns for c in goal_cols):
        goal_positions = df[goal_cols].values
    else:
        # Use current positions as goal if not available
        goal_positions = positions.copy()

    # Current proxy: STS3215 signed load (default) or raw current registers
    current_cols = [f'{current_source}{i}' for i in range(1, 7)]
    currents = df[current_cols].values if all(c in df.columns for c in current_cols) else np.zeros((len(df), 6))
    if current_lowpass_alpha > 0:
        currents = currents.astype(np.float64).copy()
        for t in range(1, len(currents)):
            currents[t] = current_lowpass_alpha * currents[t] + (1.0 - current_lowpass_alpha) * currents[t - 1]

    # Velocity (encoder steps/s)
    vel_cols = [f'vel{i}' for i in range(1, 7)]
    velocities = df[vel_cols].values if all(c in df.columns for c in vel_cols) else np.zeros((len(df), 6))

    # Voltage (decivolts)
    volt_cols = [f'volts{i}' for i in range(1, 7)]
    volts = df[volt_cols].values if all(c in df.columns for c in volt_cols) else np.zeros((len(df), 6))

    # Temperature
    temp_cols = [f'temp{i}' for i in range(1, 7)]
    temps = df[temp_cols].values if all(c in df.columns for c in temp_cols) else np.zeros((len(df), 6))

    # Timestamps
    timestamps = df['timestamp'].values if 'timestamp' in df.columns else np.arange(len(df)) * 0.01605

    # Force data (force_x, force_y, force_z)
    force_cols = ['force_x', 'force_y', 'force_z']
    if all(c in df.columns for c in force_cols):
        forces = df[force_cols].values.copy().astype(np.float64)
        # Handle -999 sentinel value (means "no force data" -> convert to 0)
        forces[forces == -999] = 0.0
        has_force = True
    else:
        forces = np.zeros((len(df), 3))
        has_force = False

    return {
        'positions': positions,
        'goal_positions': goal_positions,
        'currents': currents,
        'velocities': velocities,
        'volts': volts,
        'temps': temps,
        'timestamps': timestamps,
        'forces': forces,
        'has_force': has_force,
        'n_samples': len(df)
    }


def build_features(goal_pos, pos, current, vel, volts, temp):
    """Build 42D feature vector for SO-101 version.

    42D Feature Vector:
    - 0-5:   goal_pos1-6 (rad)
    - 6-11:  pos1-6 (rad)
    - 12-17: current1-6 (STS3215 load counts by default)
    - 18-23: vel1-6 (encoder steps/s)
    - 24-29: volts1-6 (decivolts)
    - 30-35: temp1-6 (C)
    - 36-41: pos_error1-6 = goal_pos - pos (rad)
    """
    pos_error = goal_pos - pos

    features = np.concatenate([
        goal_pos,       # 0-5
        pos,            # 6-11
        current,        # 12-17
        vel,            # 18-23
        volts,          # 24-29
        temp,           # 30-35
        pos_error,      # 36-41
    ])
    return features


def build_features_jax(goal_pos, pos, current, vel, volts, temp):
    """Build 42D feature vector for SO-101 version (JAX version)."""
    pos_error = goal_pos - pos

    features = jnp.concatenate([
        goal_pos,       # 0-5
        pos,            # 6-11
        current,        # 12-17
        vel,            # 18-23
        volts,          # 24-29
        temp,           # 30-35
        pos_error,      # 36-41
    ])
    return features


def _get_cache_key(config, mj_model, max_len, norm_stats=None):
    """Generate cache key based on config, model, and data dimensions."""
    EVAL_LOGIC_VERSION = "so101_v1"
    norm_tag = 'nonorm'
    if norm_stats is not None:
        norm_tag = hashlib.md5(np.asarray(norm_stats[0]).tobytes() + np.asarray(norm_stats[1]).tobytes()).hexdigest()[:12]
    key_parts = [
        EVAL_LOGIC_VERSION,  # Invalidates cache when eval logic changes
        str(config['history_length']),
        str(config['data_dt']),
        str(config['sim_step_size']),
        str(mj_model.nq),
        str(mj_model.nv),
        str(mj_model.nu),
        str(max_len),  # Include max_len since it affects JIT compilation
        norm_tag,
    ]
    return hashlib.md5('|'.join(key_parts).encode()).hexdigest()


def _build_eval_function(model, mjx_model, mjx_data_single, config, max_len, norm_stats=None):
    """Build JIT-compiled evaluation function with params as explicit argument."""
    history_length = config['history_length']
    data_dt = config['data_dt']
    sim_step_size = config['sim_step_size']
    feature_dim = config.get('feature_dim', FEATURE_DIM)
    # Residual torque mode
    use_residual_torque = bool(config.get('use_residual_torque', False))
    torque_constant = float(config.get('torque_constant', 0.0))

    # Training-side stability clamps: mirror them at eval
    torque_clip = float(config.get('torque_clip', 3.0))
    qvel_clip = float(config.get('qvel_clip', 0.0))

    # Feature normalization (must mirror training exactly)
    if norm_stats is not None:
        norm_mean = jnp.array(np.asarray(norm_stats[0]))
        norm_std = jnp.array(np.asarray(norm_stats[1]))

    def normalize_feat(x):
        if norm_stats is None:
            return x
        return jnp.clip((x - norm_mean) / norm_std, -10.0, 10.0)

    def single_step(carry, inputs):
        """Single simulation step for one task."""
        qpos, qvel, history, params = carry
        csv_feat, step_idx, traj_len = inputs

        # Valid when step_idx < traj_len - 1 (need next step for GT)
        valid = step_idx < (traj_len - 1)

        sim_pos = qpos[:N_JOINTS]
        sim_vel = qvel[:N_JOINTS]

        current_feat = csv_feat
        current_feat = current_feat.at[6:12].set(sim_pos)
        current_feat = current_feat.at[18:24].set(sim_vel * VEL_COUNTS_PER_RAD)

        # Update pos_error at indices 36-41
        pos_error = current_feat[0:6] - sim_pos
        current_feat = current_feat.at[36:42].set(pos_error)

        hist_flat = history.reshape(-1)

        net_feat = normalize_feat(current_feat)
        pred_tau, final_force, raw_force, gate, condition_pred, _ = model.apply(
            params, hist_flat[None, :], net_feat[None, :],
            None, ts=data_dt, training=False
        )

        # Residual torque mode: final_torque = base_torque + network_output
        # Current-source values are at indices 12-17 in csv_feat (load1-6 by default)
        # Only apply residual to arm joints (0-4), jaw (5) uses direct prediction
        if use_residual_torque:
            current_values = csv_feat[12:17]  # load1-5 in signed counts (arm only)
            base_torque = (current_values / 1000.0) * torque_constant
            # Arm: base_torque + residual, Jaw: direct prediction
            tau = jnp.concatenate([base_torque + pred_tau[0, :5], pred_tau[0, 5:6]])
        else:
            tau = pred_tau[0]
        tau_limit = jnp.full(N_JOINTS, torque_clip)
        tau = jnp.clip(tau, -tau_limit, tau_limit)

        mjx_d = mjx_data_single.replace(
            qpos=qpos,
            qvel=qvel,
            ctrl=tau.astype(jnp.float32)
        )

        def sim_body(i, d):
            d_new = mjx.step(mjx_model, d)
            d_new = d_new.replace(
                qpos=d_new.qpos.astype(jnp.float32),
                qvel=d_new.qvel.astype(jnp.float32)
            )
            return d_new

        mjx_d = jax.lax.fori_loop(0, sim_step_size, sim_body, mjx_d)

        new_qpos = mjx_d.qpos.astype(jnp.float32)
        new_qvel = mjx_d.qvel.astype(jnp.float32)
        if qvel_clip > 0:
            new_qvel = jnp.nan_to_num(jnp.clip(new_qvel, -qvel_clip, qvel_clip),
                                      nan=0.0, posinf=qvel_clip, neginf=-qvel_clip)

        new_history = jnp.roll(history, -1, axis=0)
        new_history = new_history.at[-1].set(net_feat)

        final_qpos = jnp.where(valid, new_qpos, qpos)
        final_qvel = jnp.where(valid, new_qvel, qvel)
        final_history = jnp.where(valid, new_history, history)

        sim_q = final_qpos[:N_JOINTS]
        cond_vals = condition_pred[0]  # (6,)
        force_pred = final_force[0]    # (3,) - gated force
        gate_val = gate[0, 0]          # scalar

        return (final_qpos, final_qvel, final_history, params), (sim_q, cond_vals, force_pred, gate_val)

    def eval_single_task(params, init_pos, init_history, csv_feats, traj_len):
        """Evaluate single task using scan.

        Starts from index 0 with zero-padded history (aligned with training).
        Returns sim positions, condition predictions, force predictions, and gate values.
        """
        qpos = jnp.zeros(mjx_model.nq, dtype=jnp.float32)
        qpos = qpos.at[:N_JOINTS].set(init_pos)
        qvel = jnp.zeros(mjx_model.nv, dtype=jnp.float32)

        n_eval_steps = max_len - 1
        step_indices = jnp.arange(n_eval_steps, dtype=jnp.int32)

        csv_feats_eval = csv_feats[0:n_eval_steps]

        init_carry = (qpos, qvel, init_history, params)
        inputs = (csv_feats_eval, step_indices, jnp.broadcast_to(traj_len, (n_eval_steps,)))

        _, (recorded_sim_q, recorded_cond, recorded_force, recorded_gate) = jax.lax.scan(single_step, init_carry, inputs)

        return recorded_sim_q, recorded_cond, recorded_force, recorded_gate

    return jax.jit(eval_single_task)


def evaluate_batch_mjx(model, params, task_data_list, config, mj_model, verbose=True, dump_dir=None, norm_stats=None):
    """
    Batch evaluation using MJX (GPU-accelerated, JIT-compiled).

    JIT compilation is cached across evaluation cycles for efficiency.

    Args:
        model: Neural actuator model
        params: Model parameters
        task_data_list: List of (task_name, data_dict) tuples
        config: Training config dict
        mj_model: MuJoCo model
        verbose: Print debug information

    Returns:
        dict: Results for each task with per-joint MAE at each window size
    """
    global _eval_cache
    import time
    total_start = time.time()

    history_length = config['history_length']
    data_dt = config['data_dt']
    sim_step_size = config['sim_step_size']
    feature_dim = config.get('feature_dim', FEATURE_DIM)

    # Find max trajectory length
    max_len = max(d['n_samples'] for _, d in task_data_list)
    n_tasks = len(task_data_list)

    # Check cache validity (includes max_len since it affects JIT trace)
    cache_key = _get_cache_key(config, mj_model, max_len, norm_stats)
    cache_valid = (
        _eval_cache['cache_key'] == cache_key and
        _eval_cache['mjx_model'] is not None and
        _eval_cache['eval_fn'] is not None
    )

    if verbose:
        print("\n" + "=" * 60)
        print("[MJX Batch Evaluation] Configuration:")
        print("=" * 60)
        print(f"  history_length: {history_length}")
        print(f"  data_dt: {data_dt:.4f}s")
        print(f"  sim_step_size: {sim_step_size} (from config, aligned with training)")
        print(f"  n_tasks: {n_tasks}")
        print(f"  JIT cache: {'HIT' if cache_valid else 'MISS'}")

    if cache_valid:
        mjx_model = _eval_cache['mjx_model']
        mjx_data_single = _eval_cache['mjx_data_single']
        eval_single_task_jit = _eval_cache['eval_fn']
        if verbose:
            print("\n[Step 1] Using cached MJX model and JIT function")
    else:
        # Convert MuJoCo model to MJX
        if verbose:
            print("\n[Step 1] Converting MuJoCo model to MJX...")
            t0 = time.time()
        mjx_model = mjx.put_model(mj_model)
        mj_data_template = mujoco.MjData(mj_model)
        mjx_data_single = mjx.put_data(mj_model, mj_data_template)
        if verbose:
            print(f"  Done in {time.time() - t0:.2f}s")

        # Build and JIT compile eval function
        if verbose:
            print("\n[Step 2] Building JIT-compiled evaluation function...")
            t0 = time.time()
        eval_single_task_jit = _build_eval_function(model, mjx_model, mjx_data_single, config, max_len, norm_stats)
        if verbose:
            print(f"  Done in {time.time() - t0:.2f}s")

        # Update cache
        _eval_cache['mjx_model'] = mjx_model
        _eval_cache['mjx_data_single'] = mjx_data_single
        _eval_cache['eval_fn'] = eval_single_task_jit
        _eval_cache['cache_key'] = cache_key

    # Prepare data arrays
    if verbose:
        print(f"\n[Step 3] Preparing data arrays...")
        print(f"  max_len: {max_len}")
        for task_name, data in task_data_list:
            print(f"    {task_name}: {data['n_samples']} samples")

    # Prepare batched data arrays (n_tasks, max_len, ...)
    all_csv_features = []  # (n_tasks, max_len, feature_dim)
    all_target_pos = []    # (n_tasks, max_len, 6)
    all_target_force = []  # (n_tasks, max_len, 3)
    all_init_pos = []      # (n_tasks, 6)
    all_lengths = []       # (n_tasks,)
    all_has_force = []     # (n_tasks,) - whether each task has force data

    for task_name, data in task_data_list:
        n_samples = data['n_samples']
        all_lengths.append(n_samples)
        all_has_force.append(data.get('has_force', False))

        # Build CSV features for entire trajectory
        csv_feats = np.zeros((max_len, feature_dim), dtype=np.float32)
        target_pos = np.zeros((max_len, N_JOINTS), dtype=np.float32)
        target_force = np.zeros((max_len, 3), dtype=np.float32)

        for i in range(n_samples):
            goal_pos = data['goal_positions'][i]
            pos = data['positions'][i]
            current = data['currents'][i]
            vel = data['velocities'][i]
            volts = data['volts'][i]
            temp = data['temps'][i]

            csv_feats[i] = build_features(goal_pos, pos, current, vel, volts, temp)

            if i < n_samples - 1:
                # Target is next position
                target_pos[i] = data['positions'][i + 1]
                # Target force (current step force as GT)
                target_force[i] = data['forces'][i]

        all_csv_features.append(csv_feats)
        all_target_pos.append(target_pos)
        all_target_force.append(target_force)

        # Initial position (at index 0, aligned with training zero-padding)
        init_pos = data['positions'][0].astype(np.float32)
        all_init_pos.append(init_pos)

        # Debug: verify first task's init position
        if len(all_init_pos) == 1 and verbose:
            print(f"  [DEBUG] First task init pos (index 0): {np.rad2deg(init_pos)}")

    # Convert to JAX arrays
    if verbose:
        print(f"\n[Step 3] Converting to JAX arrays...")
        t0 = time.time()
    csv_features_jax = jnp.array(np.stack(all_csv_features))  # (n_tasks, max_len, feature_dim)
    target_pos_jax = jnp.array(np.stack(all_target_pos))      # (n_tasks, max_len, 6)
    target_force_jax = jnp.array(np.stack(all_target_force))  # (n_tasks, max_len, 3)
    init_pos_jax = jnp.array(np.stack(all_init_pos))          # (n_tasks, 6)
    lengths_jax = jnp.array(all_lengths)                      # (n_tasks,)
    if verbose:
        print(f"  csv_features_jax: {csv_features_jax.shape}")
        print(f"  target_pos_jax: {target_pos_jax.shape}")
        print(f"  target_force_jax: {target_force_jax.shape}")
        print(f"  init_pos_jax: {init_pos_jax.shape}")
        print(f"  Done in {time.time() - t0:.2f}s")

    # Build initial history buffers (zero-padding, aligned with training)
    if verbose:
        print(f"\n[Step 4] Building history buffers (zero-padding)...")
        t0 = time.time()
    init_history = np.zeros((n_tasks, history_length, feature_dim), dtype=np.float32)

    init_history_jax = jnp.array(init_history)  # (n_tasks, history_length, feature_dim)
    if verbose:
        print(f"  Done in {time.time() - t0:.2f}s")

    # JIT warmup (only needed if cache was missed)
    if not cache_valid:
        if verbose:
            print(f"\n[Step 5] JIT compilation warmup...")
            t0 = time.time()
        else:
            print("  Warming up JIT...", end=" ", flush=True)
        _ = eval_single_task_jit(
            params,
            init_pos_jax[0],
            init_history_jax[0],
            csv_features_jax[0],
            lengths_jax[0]
        )
        jax.block_until_ready(_)
        if verbose:
            print(f"  JIT warmup completed in {time.time() - t0:.2f}s")
        else:
            print("done")

    # Evaluate each task
    all_results = {}
    window_sizes = [100, 300, 500]

    if verbose:
        print(f"\n[Step 6] Evaluating {n_tasks} tasks...")
        eval_start = time.time()

    for t_idx, (task_name, data) in enumerate(task_data_list):
        task_start = time.time()
        if verbose:
            print(f"  [{t_idx+1}/{n_tasks}] {task_name} ({data['n_samples']} samples)...", end=" ", flush=True)
        else:
            print(f"  [{t_idx+1}/{n_tasks}] {task_name}...", end=" ", flush=True)

        # Run evaluation (pass params as first argument)
        recorded_sim_q, recorded_cond, recorded_force, recorded_gate = eval_single_task_jit(
            params,
            init_pos_jax[t_idx],
            init_history_jax[t_idx],
            csv_features_jax[t_idx],
            lengths_jax[t_idx]
        )
        jax.block_until_ready(recorded_sim_q)
        recorded_sim_q = np.array(recorded_sim_q)
        recorded_cond = np.array(recorded_cond)
        recorded_force = np.array(recorded_force)
        recorded_gate = np.array(recorded_gate)

        # Get actual length (excluding padding)
        actual_len = min(data['n_samples'] - 1, len(recorded_sim_q))
        recorded_sim_q = recorded_sim_q[:actual_len]
        recorded_cond = recorded_cond[:actual_len]
        recorded_force = recorded_force[:actual_len]
        recorded_gate = recorded_gate[:actual_len]

        # Get force GT for this task
        target_force_task = np.array(target_force_jax[t_idx, :actual_len])
        has_force_data = all_has_force[t_idx]

        # Build GT positions (starting from index 0)
        recorded_gt_q = data['positions'][1:actual_len + 1].astype(np.float32)

        # Optionally dump rollout trajectories (for rendering / analysis)
        if dump_dir is not None:
            os.makedirs(dump_dir, exist_ok=True)
            np.savez(os.path.join(dump_dir, f'{task_name}_rollout.npz'),
                     sim_q=recorded_sim_q, gt_q=recorded_gt_q,
                     force_pred=recorded_force, force_gt=target_force_task,
                     gate=recorded_gate, cond=recorded_cond)

        # Compute MAE at each window
        results = {'n_steps': actual_len}

        for window in window_sizes:
            if window > actual_len:
                continue

            sim_q_window = recorded_sim_q[:window]
            gt_q_window = recorded_gt_q[:window]

            joint_errors_deg = np.rad2deg(sim_q_window - gt_q_window)
            mae_joints = np.mean(np.abs(joint_errors_deg), axis=0)

            for j in range(N_JOINTS):
                results[f'J{j+1}@{window}'] = float(mae_joints[j])

            # Force MAE at this window (if has force data)
            if has_force_data:
                force_pred_window = recorded_force[:window]
                force_gt_window = target_force_task[:window]
                force_error = force_pred_window - force_gt_window
                mae_force = np.mean(np.abs(force_error))
                mae_force_z = np.mean(np.abs(force_error[:, 2]))  # Only z-axis (gravity)
                results[f'Force@{window}'] = float(mae_force)
                results[f'ForceZ@{window}'] = float(mae_force_z)

        # Full trajectory MAE
        joint_errors_deg = np.rad2deg(recorded_sim_q - recorded_gt_q)
        mae_joints = np.mean(np.abs(joint_errors_deg), axis=0)
        for j in range(N_JOINTS):
            results[f'J{j+1}'] = float(mae_joints[j])

        # Force MAE (full trajectory)
        if has_force_data:
            force_error = recorded_force - target_force_task
            mae_force_all = np.mean(np.abs(force_error))
            mae_force_x = np.mean(np.abs(force_error[:, 0]))
            mae_force_z = np.mean(np.abs(force_error[:, 2]))

            results['Force'] = float(mae_force_all)
            results['ForceX'] = float(mae_force_x)
            results['ForceZ'] = float(mae_force_z)
            results['ForceAll'] = float(mae_force_all)
            results['has_force'] = True
        else:
            results['has_force'] = False

        all_results[task_name] = results

        task_time = time.time() - task_start
        force_str = f" Force={results.get('Force', 0):.2f}N" if has_force_data else ""
        joint_str = " ".join([f"J{j+1}={results[f'J{j+1}']:.1f}°" for j in range(N_JOINTS)])
        print(f"@{actual_len}: {joint_str}{force_str} ({task_time:.2f}s)")

    if verbose:
        total_time = time.time() - total_start
        eval_time = time.time() - eval_start
        print(f"\n[Summary]")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Evaluation time: {eval_time:.2f}s")
        print(f"  Avg per task: {eval_time / n_tasks:.2f}s")

    return all_results


def main():
    parser = argparse.ArgumentParser(description='Evaluate Neural Actuator (SO-101) on test sets')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--output', type=str, default='outputs/evaluation_results_so101.json', help='Output JSON file')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--dump_rollout', type=str, default=None, help='Directory to save per-task rollout trajectories (npz)')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Load model
    print(f"Loading model from {args.model_path}...")
    model, params, history_length, feature_dim, norm_stats = load_model(args.model_path, config)
    if norm_stats is not None:
        print("  Checkpoint carries feature normalization stats, applying at network input")

    # Load MuJoCo model
    mjcf_path = config.get('mjcf_path', 'robot_so101/so101_torque_scene.xml')
    print(f"Loading MuJoCo model from {mjcf_path}...")
    mj_model = mujoco.MjModel.from_xml_path(mjcf_path)

    # CRITICAL: Set timestep and solver to match training!
    data_dt = config['data_dt']
    sim_step_size = config['sim_step_size']
    sim_timestep = data_dt / sim_step_size
    mj_model.opt.timestep = sim_timestep
    # Solver settings must match training for consistent behavior!
    mj_model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    mj_model.opt.iterations = 1
    mj_model.opt.ls_iterations = 0
    mj_model.opt.tolerance = 0
    mj_model.opt.ls_tolerance = 0
    mj_model.opt.noslip_iterations = 0
    mj_model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    print(f"  Set timestep: {sim_timestep:.6f}s (data_dt={data_dt:.4f}s / sim_step_size={sim_step_size})")
    print(f"  Solver: NEWTON, contact disabled (aligned with training)")

    # Get test datasets
    test_datasets = config.get('test_datasets', {})
    if not test_datasets and 'task_dirs' in config:
        test_datasets = {}
        for task_dir in config['task_dirs']:
            task_name = "_".join(os.path.normpath(task_dir).split(os.sep)[-2:])
            for csv_path in sorted(glob.glob(os.path.join(task_dir, 'test', '*.csv'))):
                test_datasets[task_name] = csv_path

    if not test_datasets:
        raise ValueError("test_datasets is empty in config!")

    window_sizes = [100, 300, 500]

    # Check which CSV files exist
    valid_tasks = []
    for task_name, csv_path in test_datasets.items():
        if os.path.exists(csv_path):
            valid_tasks.append((task_name, csv_path))
        else:
            print(f"WARNING: {csv_path} not found, skipping {task_name}")

    if not valid_tasks:
        print("No valid test datasets found!")
        return

    # Load all task data
    print("\nLoading task data...")
    current_source = config.get('current_source', 'load')
    task_data_list = []
    for task_name, csv_path in valid_tasks:
        data = load_csv_data(csv_path, current_source, float(config.get("current_lowpass_alpha", 0.0)))
        task_data_list.append((task_name, data))

    # Run MJX batch evaluation
    results = evaluate_batch_mjx(model, params, task_data_list, config, mj_model, verbose=args.verbose,
                                 dump_dir=args.dump_rollout, norm_stats=norm_stats)

    # Compute average across all tasks at each window size
    task_results = {k: v for k, v in results.items() if k != 'AVERAGE'}

    if task_results:
        avg_results = {'n_steps': 1000}

        # Average for each window size
        for window in window_sizes:
            key_j1 = f'J1@{window}'
            if key_j1 in list(task_results.values())[0]:
                for j in range(N_JOINTS):
                    avg_results[f'J{j+1}@{window}'] = np.mean(
                        [r[f'J{j+1}@{window}'] for r in task_results.values() if f'J{j+1}@{window}' in r])
                force_vals = [r[f'Force@{window}'] for r in task_results.values() if f'Force@{window}' in r]
                if force_vals:
                    avg_results[f'Force@{window}'] = np.mean(force_vals)

        # Average for full trajectory
        for j in range(N_JOINTS):
            avg_results[f'J{j+1}'] = np.mean([r[f'J{j+1}'] for r in task_results.values()])
        force_vals = [r['Force'] for r in task_results.values() if r.get('has_force', False)]
        if force_vals:
            avg_results['Force'] = np.mean(force_vals)

        results['AVERAGE'] = avg_results

        print("\n" + "=" * 90)
        print("AVERAGE MAE across all tasks (by window size):")
        print("=" * 90)
        header = f"{'Window':<10} " + " ".join([f"{'J' + str(j+1) + ' (deg)':<10}" for j in range(N_JOINTS)]) + f"{'Force (N)':<10}"
        print(header)
        print("-" * 90)
        for window in window_sizes:
            key = f'J1@{window}'
            if key in avg_results:
                row = f"{window:<10} " + " ".join([f"{avg_results[f'J{j+1}@{window}']:<10.2f}" for j in range(N_JOINTS)])
                row += f"{avg_results.get(f'Force@{window}', 0.0):<10.2f}"
                print(row)
        print("-" * 90)
        row = f"{'Full':<10} " + " ".join([f"{avg_results[f'J{j+1}']:<10.2f}" for j in range(N_JOINTS)])
        row += f"{avg_results.get('Force', 0.0):<10.2f}"
        print(row)

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # Generate LaTeX table
    data_dt = config['data_dt']
    generate_latex_table(results, args.output.replace('.json', '_table.tex'), data_dt=data_dt)


def generate_latex_table(results, output_path, data_dt=0.01605):
    """Generate LaTeX table from results with different prediction horizons."""
    window_sizes = [100, 300, 500]

    # Filter to windows that exist in results
    avg = results.get('AVERAGE', {})
    available_windows = [w for w in window_sizes if f'J1@{w}' in avg]

    if not available_windows:
        print("No window-based results found, skipping LaTeX table generation.")
        return

    # Calculate times for each window
    times = {w: w * data_dt for w in available_windows}

    # Number of columns for horizons
    n_horizons = len(available_windows)

    latex = []
    latex.append(r"\begin{table}[t]")
    latex.append(r"  \centering")
    latex.append(r"  \small")
    latex.append(r"  \setlength{\tabcolsep}{5pt}")
    latex.append(r"  \caption{Simulation accuracy across different time horizons on the SO-101 arm. Values show mean absolute error on the test set.}")
    latex.append(r"  \label{tab:sim-accuracy-s101}")
    latex.append(r"  \resizebox{0.95\linewidth}{!}{%")

    # Build column spec
    col_spec = "lcc" + "c" * n_horizons
    latex.append(f"  \\begin{{tabular}}{{{col_spec}}}")
    latex.append(r"    \toprule")

    # Header row 1: multirow and multicolumn
    latex.append(r"    \multirow{2}{*}{\textbf{Joint}} &")
    latex.append(r"    \multirow{2}{*}{\textbf{Type}} &")
    latex.append(r"    \multirow{2}{*}{\textbf{Unit}} &")
    latex.append(f"    \\multicolumn{{{n_horizons}}}{{c}}{{\\textbf{{Prediction Horizon}}}} \\\\")

    # cmidrule
    latex.append(f"    \\cmidrule(lr){{4-{3 + n_horizons}}}")

    # Header row 2: time values
    time_strs = [f"{times[w]:.2f}s" for w in available_windows]
    latex.append(f"     & & & " + " & ".join(time_strs) + r" \\")

    # Header row 3: step counts
    step_strs = [f"({w} steps)" for w in available_windows]
    latex.append(f"     & & & " + " & ".join(step_strs) + r" \\")

    latex.append(r"    \midrule")

    # Data rows for J1-J6 (all revolute)
    for j in range(1, N_JOINTS + 1):
        joint_name = f"Joint{j}"
        values = [f"{avg.get(f'J{j}@{w}', 0):.2f}" for w in available_windows]
        latex.append(f"    {joint_name} & Revolute & deg & " + " & ".join(values) + r" \\")

    latex.append(r"    \bottomrule")
    latex.append(r"  \end{tabular}%")
    latex.append(r"  }")
    latex.append(r"\end{table}")

    with open(output_path, 'w') as f:
        f.write('\n'.join(latex))
    print(f"LaTeX table saved to {output_path}")


if __name__ == "__main__":
    main()
