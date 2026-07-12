"""
Evaluate Neural Actuator on test sets.

Computes per-joint MAE for each task and outputs results in JSON format.
Joint 1-4: Revolute joints (degrees)
Joint 5-6: Prismatic joints (mm) - gripper fingers
"""

import argparse
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
    'max_len': None,  # Track max_len for cache invalidation
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

    # Feature dimension (36D with goal_aperture and gripper_error)
    feature_dim = config.get('feature_dim', 36)

    # Create model using factory function
    model = create_model(
        model_type=model_type,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        dropout_rate=dropout_rate,
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


def load_csv_data(csv_path, current_lowpass_alpha=0.0):
    """Load CSV data and extract relevant columns."""
    df = pd.read_csv(csv_path)

    # Joint positions (pos1-5)
    pos_cols = [f'pos{i}' for i in range(1, 6)]
    positions = df[pos_cols].values

    # Goal positions (target)
    goal_cols = [f'goal_pos{i}' for i in range(1, 6)]
    if all(c in df.columns for c in goal_cols):
        goal_positions = df[goal_cols].values
    else:
        # Use current positions as goal if not available
        goal_positions = positions.copy()

    # Aperture
    if 'aperture' in df.columns:
        aperture = df['aperture'].values / 1000.0  # Convert mm to meters
    else:
        aperture = np.zeros(len(df))

    # Current
    current_cols = [f'current{i}' for i in range(1, 6)]
    currents = df[current_cols].values if all(c in df.columns for c in current_cols) else np.zeros((len(df), 5))
    if current_lowpass_alpha > 0:
        currents = currents.astype(np.float64).copy()
        for t in range(1, len(currents)):
            currents[t] = current_lowpass_alpha * currents[t] + (1.0 - current_lowpass_alpha) * currents[t - 1]

    # Velocity
    vel_cols = [f'vel{i}' for i in range(1, 6)]
    velocities = df[vel_cols].values if all(c in df.columns for c in vel_cols) else np.zeros((len(df), 5))

    # Voltage
    volt_cols = [f'volts{i}' for i in range(1, 6)]
    volts = df[volt_cols].values if all(c in df.columns for c in volt_cols) else np.zeros((len(df), 5))

    # Temperature
    temp_cols = [f'temp{i}' for i in range(1, 6)]
    temps = df[temp_cols].values if all(c in df.columns for c in temp_cols) else np.zeros((len(df), 5))

    # Goal aperture (for 36D features)
    if 'goal_aperture' in df.columns:
        goal_aperture = df['goal_aperture'].values  # Already in mm
    else:
        # Use current aperture as goal if not available
        goal_aperture = df['aperture'].values if 'aperture' in df.columns else np.zeros(len(df))

    # Timestamps
    timestamps = df['timestamp'].values if 'timestamp' in df.columns else np.arange(len(df)) * 0.016

    # Force data (force_x, force_y, force_z)
    force_cols = ['force_x', 'force_y', 'force_z']
    if all(c in df.columns for c in force_cols):
        forces = df[force_cols].values.copy()
        # Handle -999 sentinel value (means "no force data" -> convert to 0)
        forces[forces == -999] = 0.0
        has_force = True
    else:
        forces = np.zeros((len(df), 3))
        has_force = False

    return {
        'positions': positions,
        'goal_positions': goal_positions,
        'aperture': aperture,
        'goal_aperture': goal_aperture,
        'currents': currents,
        'velocities': velocities,
        'volts': volts,
        'temps': temps,
        'timestamps': timestamps,
        'forces': forces,
        'has_force': has_force,
        'n_samples': len(df)
    }


def build_features(goal_pos, pos, aperture, current, vel, volts, temp, goal_aperture=None):
    """Build 36D feature vector.

    36D Feature Vector:
    - 0-4: goal_pos (5)
    - 5-8: pos[:4] (4)
    - 9: aperture (1)
    - 10-14: current (5)
    - 15-19: vel (5)
    - 20-24: volts (5)
    - 25-29: temp (5)
    - 30: goal_aperture (1)
    - 31-34: arm_error = goal_pos[:4] - pos[:4] (4)
    - 35: gripper_error = goal_aperture - aperture (1)
    """
    if goal_aperture is None:
        goal_aperture = aperture

    arm_error = goal_pos[:4] - pos[:4]
    gripper_error = goal_aperture - aperture

    features = np.concatenate([
        goal_pos,           # 0-4
        pos[:4],            # 5-8
        [aperture],         # 9
        current,            # 10-14
        vel,                # 15-19
        volts,              # 20-24
        temp,               # 25-29
        [goal_aperture],    # 30
        arm_error,          # 31-34
        [gripper_error],    # 35
    ])
    return features


def build_features_jax(goal_pos, pos, aperture, current, vel, volts, temp, goal_aperture=None):
    """Build 36D feature vector (JAX version)."""
    if goal_aperture is None:
        goal_aperture = aperture

    arm_error = goal_pos[:4] - pos[:4]
    gripper_error = goal_aperture - aperture

    features = jnp.concatenate([
        goal_pos,                   # 0-4
        pos[:4],                    # 5-8
        jnp.array([aperture]),      # 9
        current,                    # 10-14
        vel,                        # 15-19
        volts,                      # 20-24
        temp,                       # 25-29
        jnp.array([goal_aperture]), # 30
        arm_error,                  # 31-34
        jnp.array([gripper_error]), # 35
    ])
    return features


def _get_cache_key(config, mj_model, max_len, norm_stats=None):
    """Generate cache key based on config, model, and data dimensions."""
    EVAL_LOGIC_VERSION = "v1"
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
    # Feature dimension (36D with goal_aperture and gripper_error)
    feature_dim = config.get('feature_dim', 36)
    # Residual torque mode
    use_residual_torque = config['use_residual_torque']
    torque_constant = float(config['torque_constant'])

    # Stability clamps used in training are mirrored at eval when present in the config
    gripper_torque_clip = config.get('gripper_torque_clip', None)
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

        sim_pos = qpos[:5]
        sim_vel = qvel[:5]
        aperture_val = qpos[4] * 1000.0  # Convert m to mm

        current_feat = csv_feat
        current_feat = current_feat.at[5:9].set(sim_pos[:4])
        current_feat = current_feat.at[9].set(aperture_val)
        current_feat = current_feat.at[15:20].set(sim_vel)

        # Update arm_error at indices 31-34 (36D format)
        arm_error = current_feat[0:4] - sim_pos[:4]
        current_feat = current_feat.at[31:35].set(arm_error)
        # Update gripper_error at index 35: goal_aperture (index 30) - aperture (index 9)
        gripper_error = current_feat[30] - aperture_val
        current_feat = current_feat.at[35].set(gripper_error)

        hist_flat = history.reshape(-1)

        net_feat = normalize_feat(current_feat)
        pred_tau, final_force, raw_force, gate, condition_pred, _ = model.apply(
            params, hist_flat[None, :], net_feat[None, :],
            None, ts=data_dt, training=False
        )

        # Residual torque mode: final_torque = base_torque + network_output
        # Current values are at indices 10-14 in csv_feat (current1-5)
        # Only apply residual to arm joints (0-3), gripper (4) uses direct prediction
        if use_residual_torque:
            current_values = csv_feat[10:14]  # current1-4 in mA (arm only)
            base_torque = (current_values / 1000.0) * torque_constant
            # Arm: base_torque + residual, Gripper: direct prediction
            tau = jnp.concatenate([base_torque + pred_tau[0, :4], pred_tau[0, 4:5]])
        else:
            tau = pred_tau[0]

        if gripper_torque_clip is not None:
            tau_limit = jnp.array([5.0, 5.0, 5.0, 5.0, float(gripper_torque_clip)])
            tau = jnp.clip(tau, -tau_limit, tau_limit)

        mjx_d = mjx_data_single.replace(
            qpos=qpos,
            qvel=qvel,
            ctrl=jnp.zeros(mjx_model.nu, dtype=jnp.float32).at[:5].set(tau)
        )

        def sim_body(i, d):
            d_new = mjx.step(mjx_model, d)
            d_new = d_new.replace(
                qpos=d_new.qpos.astype(jnp.float32),
                qvel=d_new.qvel.astype(jnp.float32)
            )
            return d_new

        mjx_d = jax.lax.fori_loop(0, sim_step_size, sim_body, mjx_d)

        new_qpos = mjx_d.qpos.at[4].set(jnp.clip(mjx_d.qpos[4], -0.011, 0.02))
        new_qpos = new_qpos.at[5].set(jnp.clip(new_qpos[5], -0.011, 0.02))
        new_qpos = new_qpos.astype(jnp.float32)
        new_qvel = mjx_d.qvel.astype(jnp.float32)
        if qvel_clip > 0:
            new_qvel = jnp.nan_to_num(jnp.clip(new_qvel, -qvel_clip, qvel_clip),
                                      nan=0.0, posinf=qvel_clip, neginf=-qvel_clip)

        new_history = jnp.roll(history, -1, axis=0)
        new_history = new_history.at[-1].set(net_feat)

        final_qpos = jnp.where(valid, new_qpos, qpos)
        final_qvel = jnp.where(valid, new_qvel, qvel)
        final_history = jnp.where(valid, new_history, history)

        sim_q = final_qpos[:5]
        # Extract condition prediction (5,) - one per motor
        cond_vals = condition_pred[0]  # (5,)
        # Extract force prediction (3,) and gate (1,)
        force_pred = final_force[0]  # (3,) - gated force
        gate_val = gate[0, 0]  # scalar

        return (final_qpos, final_qvel, final_history, params), (sim_q, cond_vals, force_pred, gate_val)

    def eval_single_task(params, init_pos, init_history, csv_feats, traj_len):
        """Evaluate single task using scan.

        Starts from index 0 with zero-padded history (same as training).
        Returns sim positions, condition predictions, force predictions, and gate values.
        """
        qpos = jnp.zeros(mjx_model.nq, dtype=jnp.float32)
        qpos = qpos.at[:4].set(init_pos[:4])
        qpos = qpos.at[4].set(init_pos[4])
        qpos = qpos.at[5].set(init_pos[4])
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
    # Feature dimension (36D with goal_aperture and gripper_error)
    feature_dim = config.get('feature_dim', 36)

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
        print(f"  sim_step_size: {sim_step_size}")
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
    # Pre-compute all CSV features for each task
    all_csv_features = []  # (n_tasks, max_len, feature_dim)
    all_target_pos = []    # (n_tasks, max_len, 5)
    all_target_force = []  # (n_tasks, max_len, 3)
    all_init_pos = []      # (n_tasks, 5)
    all_lengths = []       # (n_tasks,)
    all_has_force = []     # (n_tasks,) - whether each task has force data

    for task_name, data in task_data_list:
        n_samples = data['n_samples']
        all_lengths.append(n_samples)
        all_has_force.append(data.get('has_force', False))

        # Build CSV features for entire trajectory
        csv_feats = np.zeros((max_len, feature_dim), dtype=np.float32)
        target_pos = np.zeros((max_len, 5), dtype=np.float32)
        target_force = np.zeros((max_len, 3), dtype=np.float32)

        for i in range(n_samples):
            goal_pos = data['goal_positions'][i]
            pos = data['positions'][i]
            aperture = data['aperture'][i] * 1000.0  # Convert m back to mm for feature
            current = data['currents'][i]
            vel = data['velocities'][i]
            volts = data['volts'][i]
            temp = data['temps'][i]
            goal_aperture = data['goal_aperture'][i]  # Already in mm

            csv_feats[i] = build_features(goal_pos, pos, aperture, current, vel, volts, temp, goal_aperture)

            if i < n_samples - 1:
                # Target is next position
                target_pos[i, :4] = data['positions'][i + 1, :4]
                target_pos[i, 4] = data['aperture'][i + 1]
                # Target force (current step force as GT)
                target_force[i] = data['forces'][i]

        all_csv_features.append(csv_feats)
        all_target_pos.append(target_pos)
        all_target_force.append(target_force)

        # Initial position at index 0
        init_pos = np.zeros(5, dtype=np.float32)
        init_pos[:4] = data['positions'][0, :4]
        init_pos[4] = data['aperture'][0]
        all_init_pos.append(init_pos)

    # Convert to JAX arrays
    if verbose:
        print(f"\n[Step 3] Converting to JAX arrays...")
        t0 = time.time()
    csv_features_jax = jnp.array(np.stack(all_csv_features))  # (n_tasks, max_len, feature_dim)
    target_pos_jax = jnp.array(np.stack(all_target_pos))      # (n_tasks, max_len, 5)
    target_force_jax = jnp.array(np.stack(all_target_force))  # (n_tasks, max_len, 3)
    init_pos_jax = jnp.array(np.stack(all_init_pos))          # (n_tasks, 5)
    lengths_jax = jnp.array(all_lengths)                       # (n_tasks,)
    if verbose:
        print(f"  csv_features_jax: {csv_features_jax.shape}")
        print(f"  target_pos_jax: {target_pos_jax.shape}")
        print(f"  target_force_jax: {target_force_jax.shape}")
        print(f"  init_pos_jax: {init_pos_jax.shape}")
        print(f"  Done in {time.time() - t0:.2f}s")

    # Zero-padded initial history, same as training when a rollout starts at the trajectory beginning
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
    window_sizes = [10, 100, 200, 300, 400, 500, 600]

    if verbose:
        print(f"\n[Step 6] Evaluating {n_tasks} tasks...")
        eval_start = time.time()

    # Collect all condition predictions and ground truths for classification metrics
    # Two sets: all 5 motors (aligns with training) and joint3 only (meaningful metric)
    all_cond_preds_5motors = []  # All 5 motors (for training alignment check)
    all_cond_gts_5motors = []
    all_cond_preds_j3 = []       # Joint3 only (the meaningful metric)
    all_cond_gts_j3 = []

    for t_idx, (task_name, data) in enumerate(task_data_list):
        task_start = time.time()
        if verbose:
            print(f"  [{t_idx+1}/{n_tasks}] {task_name} ({data['n_samples']} samples)...", end=" ", flush=True)
        else:
            print(f"  [{t_idx+1}/{n_tasks}] {task_name}...", end=" ", flush=True)

        # Run evaluation (pass params as first argument)
        # Returns sim positions, condition predictions, force predictions, and gate values
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

        # Determine ground truth condition from task name
        # "normal" in task name -> all motors cond_gt = [1, 1, 1, 1, 1]
        # "degrade" in task name -> only joint3 degraded cond_gt = [1, 1, 0, 1, 1]
        is_degrade = 'degrade' in task_name.lower()
        if is_degrade:
            cond_gt_5motors = np.array([1.0, 1.0, 0.0, 1.0, 1.0])
        else:
            cond_gt_5motors = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        cond_gt_j3 = 0.0 if is_degrade else 1.0

        # Collect for classification metrics
        # recorded_cond shape: (actual_len, 5)
        for step_cond in recorded_cond:
            # All 5 motors (for alignment with training)
            all_cond_preds_5motors.extend(step_cond.tolist())
            all_cond_gts_5motors.extend(cond_gt_5motors.tolist())
            # Joint3 only (the meaningful metric)
            all_cond_preds_j3.append(step_cond[2])
            all_cond_gts_j3.append(cond_gt_j3)

        # Build GT positions (starting from index 0)
        recorded_gt_q = []
        for i in range(0, actual_len):
            gt_q = data['positions'][i + 1]
            gt_aperture = data['aperture'][i + 1]
            recorded_gt_q.append(np.concatenate([gt_q, [gt_aperture, gt_aperture]]))
        recorded_gt_q = np.array(recorded_gt_q)

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

            joint_errors = sim_q_window - gt_q_window[:, :5]
            joint_errors_deg = np.rad2deg(joint_errors[:, :4])
            mae_j1_j4 = np.mean(np.abs(joint_errors_deg), axis=0)

            gripper_sim = sim_q_window[:, 4]  # Single finger position (m), same as CSV aperture
            gripper_gt = gt_q_window[:, 5]
            gripper_error_mm = (gripper_sim - gripper_gt) * 1000
            mae_j5 = np.mean(np.abs(gripper_error_mm))

            results[f'J1@{window}'] = float(mae_j1_j4[0])
            results[f'J2@{window}'] = float(mae_j1_j4[1])
            results[f'J3@{window}'] = float(mae_j1_j4[2])
            results[f'J4@{window}'] = float(mae_j1_j4[3])
            results[f'J5@{window}'] = float(mae_j5)
            results[f'J6@{window}'] = float(mae_j5)

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
        joint_errors = recorded_sim_q - recorded_gt_q[:, :5]
        joint_errors_deg = np.rad2deg(joint_errors[:, :4])
        mae_j1_j4 = np.mean(np.abs(joint_errors_deg), axis=0)
        gripper_sim = recorded_sim_q[:, 4]  # Single finger position (m), same as CSV aperture
        gripper_gt = recorded_gt_q[:, 5]
        gripper_error_mm = (gripper_sim - gripper_gt) * 1000
        mae_j5 = np.mean(np.abs(gripper_error_mm))

        results['J1'] = float(mae_j1_j4[0])
        results['J2'] = float(mae_j1_j4[1])
        results['J3'] = float(mae_j1_j4[2])
        results['J4'] = float(mae_j1_j4[3])
        results['J5'] = float(mae_j5)
        results['J6'] = float(mae_j5)

        # Force MAE (full trajectory)
        if has_force_data:
            force_error = recorded_force - target_force_task
            mae_force_all = np.mean(np.abs(force_error))
            mae_force_x = np.mean(np.abs(force_error[:, 0]))  # X-axis only
            mae_force_z = np.mean(np.abs(force_error[:, 2]))  # Z-axis only

            # For pushing_gauge: use axis-specific Force MAE based on task name
            # 'front' tasks -> X axis, 'top' tasks -> Z axis
            if 'front' in task_name.lower():
                mae_force_primary = mae_force_x
                results['ForceAxis'] = 'X'
            elif 'top' in task_name.lower():
                mae_force_primary = mae_force_z
                results['ForceAxis'] = 'Z'
            else:
                mae_force_primary = mae_force_all  # Default: use all axes
                results['ForceAxis'] = 'ALL'

            results['Force'] = float(mae_force_primary)  # Primary axis for this task
            results['ForceX'] = float(mae_force_x)
            results['ForceZ'] = float(mae_force_z)
            results['ForceAll'] = float(mae_force_all)
            results['has_force'] = True
        else:
            results['has_force'] = False

        all_results[task_name] = results

        task_time = time.time() - task_start
        force_str = f" Force={results.get('Force', 0):.2f}N" if has_force_data else ""
        print(f"@{actual_len}: J1={results['J1']:.1f}° J2={results['J2']:.1f}° J3={results['J3']:.1f}° J4={results['J4']:.1f}° Grip={results['J5']:.1f}mm{force_str} ({task_time:.2f}s)")

    if verbose:
        total_time = time.time() - total_start
        eval_time = time.time() - eval_start
        print(f"\n[Summary]")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Evaluation time: {eval_time:.2f}s")
        print(f"  Avg per task: {eval_time / n_tasks:.2f}s")

    # Compute classification metrics for both: all 5 motors (training alignment) and joint3 only (meaningful)
    def compute_classification_metrics(preds, gts, name=""):
        """Compute classification metrics. Positive class = degraded (gt=0)."""
        preds = np.array(preds)
        gts = np.array(gts)

        # Training convention: pred < 0.5 -> degraded (positive), pred >= 0.5 -> normal (negative)
        pred_degraded = (preds < 0.5).astype(float)  # 1 if predicted degraded
        gt_degraded = (gts < 0.5).astype(float)      # 1 if actual degraded

        # Confusion matrix (positive = degraded)
        tp = np.sum((pred_degraded == 1) & (gt_degraded == 1))  # Correct degraded
        tn = np.sum((pred_degraded == 0) & (gt_degraded == 0))  # Correct normal
        fp = np.sum((pred_degraded == 1) & (gt_degraded == 0))  # False alarm
        fn = np.sum((pred_degraded == 0) & (gt_degraded == 1))  # Miss detection

        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-7)
        precision = tp / (tp + fp + 1e-7)   # Of predicted degraded, how many are actually degraded
        recall = tp / (tp + fn + 1e-7)      # Of actual degraded, how many are correctly predicted (sensitivity)
        f1 = 2 * precision * recall / (precision + recall + 1e-7)
        specificity = tn / (tn + fp + 1e-7) # Of actual normal, how many are correctly predicted

        # AUC-ROC: P(pred_normal > pred_degraded) for (normal, degraded) pairs
        # Higher pred means more normal, so for correct ranking normal should have higher pred
        normal_preds = preds[gts >= 0.5]
        degraded_preds = preds[gts < 0.5]
        n_normal = len(normal_preds)
        n_degraded = len(degraded_preds)

        if n_normal > 0 and n_degraded > 0:
            all_preds_sorted_idx = np.argsort(preds)
            ranks = np.zeros_like(preds)
            ranks[all_preds_sorted_idx] = np.arange(1, len(preds) + 1)
            rank_sum_normal = np.sum(ranks[gts >= 0.5])
            auc = (rank_sum_normal - n_normal * (n_normal + 1) / 2) / (n_normal * n_degraded)
            auc = np.clip(auc, 0.0, 1.0)
        else:
            auc = 0.5

        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'specificity': float(specificity),
            'auc_roc': float(auc),
            'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
            'n_samples': len(preds),
        }

    # Compute metrics for both sets
    if all_cond_preds_5motors and len(set(all_cond_gts_5motors)) > 1:
        metrics_5motors = compute_classification_metrics(all_cond_preds_5motors, all_cond_gts_5motors, "5 Motors")
        all_results['CLASSIFICATION_5MOTORS'] = metrics_5motors

        if verbose:
            print(f"\n[Classification Metrics - All 5 Motors (aligns with training)]")
            print(f"  Accuracy:    {metrics_5motors['accuracy']*100:.1f}%")
            print(f"  Precision:   {metrics_5motors['precision']*100:.1f}%")
            print(f"  Recall:      {metrics_5motors['recall']*100:.1f}%")
            print(f"  F1-Score:    {metrics_5motors['f1']*100:.1f}%")
            print(f"  Specificity: {metrics_5motors['specificity']*100:.1f}%")
            print(f"  AUC-ROC:     {metrics_5motors['auc_roc']:.3f}")
            print(f"  Confusion:   TP={metrics_5motors['tp']}, TN={metrics_5motors['tn']}, FP={metrics_5motors['fp']}, FN={metrics_5motors['fn']}")

    if all_cond_preds_j3 and len(set(all_cond_gts_j3)) > 1:
        metrics_j3 = compute_classification_metrics(all_cond_preds_j3, all_cond_gts_j3, "Joint3")
        all_results['CLASSIFICATION_J3'] = metrics_j3

        if verbose:
            print(f"\n[Classification Metrics - Joint3 Only (meaningful metric)]")
            print(f"  Accuracy:    {metrics_j3['accuracy']*100:.1f}%")
            print(f"  Precision:   {metrics_j3['precision']*100:.1f}%")
            print(f"  Recall:      {metrics_j3['recall']*100:.1f}%")
            print(f"  F1-Score:    {metrics_j3['f1']*100:.1f}%")
            print(f"  Specificity: {metrics_j3['specificity']*100:.1f}%")
            print(f"  AUC-ROC:     {metrics_j3['auc_roc']:.3f}")
            print(f"  Confusion:   TP={metrics_j3['tp']}, TN={metrics_j3['tn']}, FP={metrics_j3['fp']}, FN={metrics_j3['fn']}")

    return all_results


def evaluate_on_csv(model, params, csv_path, config, mj_model, verbose=False):
    """
    Evaluate model on a single CSV file.
    Returns per-joint MAE (in degrees for J1-J4, mm for J5-J6).
    """
    history_length = config['history_length']
    data_dt = config['data_dt']
    sim_step_size = config['sim_step_size']
    # Feature dimension (36D with goal_aperture and gripper_error)
    feature_dim = config.get('feature_dim', 36)
    # Residual torque mode
    use_residual_torque = config['use_residual_torque']
    torque_constant = float(config['torque_constant'])

    # Load data
    data = load_csv_data(csv_path, float(config.get("current_lowpass_alpha", 0.0)))
    n_samples = data['n_samples']

    if verbose:
        print(f"  Loaded {n_samples} samples from {os.path.basename(csv_path)}")

    # Initialize MuJoCo
    mj_data = mujoco.MjData(mj_model)

    # Start from index 0 with zero-padded history (same as training)
    start_idx = 0

    # Initialize simulation state from CSV
    mj_data.qpos[:4] = data['positions'][start_idx, :4]
    mj_data.qpos[4] = data['aperture'][start_idx]
    mj_data.qpos[5] = data['aperture'][start_idx]
    mj_data.qvel[:] = 0
    mujoco.mj_forward(mj_model, mj_data)

    # Build history buffer with zero-padding
    history_buffer = []
    for i in range(history_length):
        feat = np.zeros(feature_dim, dtype=np.float32)
        history_buffer.append(feat)

    # Initialize state based on model type
    hidden_dim = config['hidden_dim']
    model_type = get_model_type_from_config(config)

    if 'h0_torque' in params.get('params', {}):
        h_torque = jnp.asarray(params['params']['h0_torque'])
        h_force = jnp.asarray(params['params'].get('h0_force', jnp.zeros((1, hidden_dim))))
    else:
        h_torque = jnp.zeros((1, hidden_dim))
        h_force = jnp.zeros((1, hidden_dim))

    # Different state structures for different model types
    if model_type == 'lstm':
        # LSTM needs ((h, c), (h, c)) structure
        c_torque = jnp.zeros_like(h_torque)
        c_force = jnp.zeros_like(h_force)
        lnn_state = ((h_torque, c_torque), (h_force, c_force))
    elif model_type in ['gru', 'lnn']:
        # GRU/LNN use (h, h) structure
        lnn_state = (h_torque, h_force)
    else:
        # MLP/Transformer don't use state
        lnn_state = None

    # Run simulation
    recorded_sim_q = []
    recorded_gt_q = []

    for csv_idx in range(start_idx, min(n_samples - 1, start_idx + 1000)):
        # Build current features from simulation state + CSV data
        sim_pos = mj_data.qpos[:5].copy()
        sim_vel = mj_data.qvel[:5].copy()
        sim_aperture = mj_data.qpos[4]

        goal_pos = data['goal_positions'][csv_idx]
        current = data['currents'][csv_idx]
        volts = data['volts'][csv_idx]
        temp = data['temps'][csv_idx]
        goal_aperture = data['goal_aperture'][csv_idx]  # Already in mm

        curr_feat = build_features(goal_pos, sim_pos, sim_aperture * 1000.0, current, sim_vel, volts, temp, goal_aperture)

        # Build history input (flattened: history_len * feature_dim)
        history_np = np.array(history_buffer[-history_length:])
        hist_flat = history_np.flatten()  # (history_len * feature_dim,)
        hist_jax = jnp.array(hist_flat)[jnp.newaxis, :]  # (1, history_len * feature_dim)
        curr_jax = jnp.array(curr_feat)[jnp.newaxis, :]  # (1, feature_dim)

        # Model forward pass (unified interface for all model types)
        # Returns 6 values: torque, final_force, raw_force, gate, condition, new_state
        pred_tau, final_force, raw_force, gate, condition, new_state = model.apply(
            params, hist_jax, curr_jax, lnn_state, ts=data_dt, training=False
        )
        lnn_state = new_state

        # Residual torque mode: final_torque = base_torque + network_output
        # Current values are at indices 10-14 in curr_feat (current1-5)
        # Only apply residual to arm joints (0-3), gripper (4) uses direct prediction
        if use_residual_torque:
            current_values = curr_feat[10:14]  # current1-4 in mA (arm only)
            base_torque = (current_values / 1000.0) * torque_constant
            # Arm: base_torque + residual, Gripper: direct prediction
            tau = np.concatenate([np.array(base_torque + pred_tau[0, :4]), np.array(pred_tau[0, 4:5])])
        else:
            tau = np.array(pred_tau[0])
        mj_data.ctrl[:5] = tau

        # Step simulation
        for _ in range(sim_step_size):
            mujoco.mj_step(mj_model, mj_data)

        # Enforce gripper limits (same range as the MJX path)
        mj_data.qpos[4] = np.clip(mj_data.qpos[4], -0.011, 0.02)
        mj_data.qpos[5] = np.clip(mj_data.qpos[5], -0.011, 0.02)

        # Update history
        history_buffer.append(curr_feat.copy())
        if len(history_buffer) > history_length:
            history_buffer.pop(0)

        # Record positions
        # GT: next CSV position (what we're trying to track)
        gt_q = data['positions'][csv_idx + 1]
        gt_aperture = data['aperture'][csv_idx + 1]

        # Sim: current simulation state
        sim_q = mj_data.qpos[:5].copy()

        recorded_sim_q.append(sim_q)
        recorded_gt_q.append(np.concatenate([gt_q, [gt_aperture, gt_aperture]]))

    recorded_sim_q = np.array(recorded_sim_q)
    recorded_gt_q = np.array(recorded_gt_q)

    # Window sizes for MAE computation
    window_sizes = [10, 100, 200, 300, 400, 500, 600]

    # Compute per-joint MAE at different window sizes
    # J1-J4: revolute joints (radians -> degrees)
    # J5-J6: prismatic joints (meters -> mm)

    results = {'n_steps': len(recorded_sim_q)}

    for window in window_sizes:
        if window > len(recorded_sim_q):
            continue

        # Use first `window` steps
        sim_q_window = recorded_sim_q[:window]
        gt_q_window = recorded_gt_q[:window]

        joint_errors = sim_q_window - gt_q_window[:, :5]

        # Convert to degrees for J1-J4
        joint_errors_deg = np.rad2deg(joint_errors[:, :4])
        mae_j1_j4 = np.mean(np.abs(joint_errors_deg), axis=0)

        # Convert to mm for J5 (gripper)
        gripper_sim = sim_q_window[:, 4]  # Single finger position (m), same as CSV aperture
        gripper_gt = gt_q_window[:, 5]  # aperture
        gripper_error_mm = (gripper_sim - gripper_gt) * 1000  # to mm
        mae_j5 = np.mean(np.abs(gripper_error_mm))

        # Store with window suffix
        results[f'J1@{window}'] = float(mae_j1_j4[0])
        results[f'J2@{window}'] = float(mae_j1_j4[1])
        results[f'J3@{window}'] = float(mae_j1_j4[2])
        results[f'J4@{window}'] = float(mae_j1_j4[3])
        results[f'J5@{window}'] = float(mae_j5)
        results[f'J6@{window}'] = float(mae_j5)  # Same as J5 (symmetric gripper)

    # Also compute overall MAE (full trajectory)
    joint_errors = recorded_sim_q - recorded_gt_q[:, :5]
    joint_errors_deg = np.rad2deg(joint_errors[:, :4])
    mae_j1_j4 = np.mean(np.abs(joint_errors_deg), axis=0)
    gripper_sim = recorded_sim_q[:, 4]  # Single finger position (m), same as CSV aperture
    gripper_gt = recorded_gt_q[:, 5]
    gripper_error_mm = (gripper_sim - gripper_gt) * 1000
    mae_j5 = np.mean(np.abs(gripper_error_mm))

    results['J1'] = float(mae_j1_j4[0])
    results['J2'] = float(mae_j1_j4[1])
    results['J3'] = float(mae_j1_j4[2])
    results['J4'] = float(mae_j1_j4[3])
    results['J5'] = float(mae_j5)
    results['J6'] = float(mae_j5)

    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate Neural Actuator on test sets')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--output', type=str, default='outputs/evaluation_results.json', help='Output JSON file')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--use_cpu', action='store_true', help='Use CPU (MuJoCo) instead of GPU (MJX)')
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
        if args.use_cpu:
            raise SystemExit("--use_cpu path does not support normalized checkpoints; use the MJX (GPU) path.")

    # Load MuJoCo model
    mjcf_path = config.get('mjcf_path', 'robot/scene.xml')
    print(f"Loading MuJoCo model from {mjcf_path}...")
    mj_model = mujoco.MjModel.from_xml_path(mjcf_path)

    # Timestep and solver must match training:
    # training sets mj_model.opt.timestep = data_dt / sim_step_size
    data_dt = config['data_dt']
    sim_step_size = config['sim_step_size']
    sim_timestep = data_dt / sim_step_size
    mj_model.opt.timestep = sim_timestep
    mj_model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    mj_model.opt.iterations = 1
    mj_model.opt.ls_iterations = 0
    mj_model.opt.tolerance = 0
    mj_model.opt.ls_tolerance = 0
    mj_model.opt.noslip_iterations = 0
    mj_model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    print(f"  Set timestep: {sim_timestep:.6f}s (data_dt={data_dt:.4f}s / sim_step_size={sim_step_size})")
    print(f"  Solver: NEWTON, contact disabled")

    # Get test datasets
    test_datasets = config['test_datasets']

    if not test_datasets:
        raise ValueError("test_datasets is empty in config!")

    window_sizes = [10, 100, 200, 300, 400, 500, 600]

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

    # Use MJX (GPU) or MuJoCo (CPU) based on flag
    if args.use_cpu:
        print("\n" + "=" * 60)
        print("Evaluating on test sets (CPU mode)...")
        print("=" * 60)

        results = {}
        for task_name, csv_path in valid_tasks:
            print(f"\nTask: {task_name}")
            task_results = evaluate_on_csv(model, params, csv_path, config, mj_model, args.verbose)
            results[task_name] = task_results
            print(f"  Full: J1={task_results['J1']:.2f}° J2={task_results['J2']:.2f}° J3={task_results['J3']:.2f}° J4={task_results['J4']:.2f}° Grip={task_results['J5']:.2f}mm")
    else:
        # Load all task data
        print("\nLoading task data...")
        task_data_list = []
        for task_name, csv_path in valid_tasks:
            data = load_csv_data(csv_path, float(config.get("current_lowpass_alpha", 0.0)))
            task_data_list.append((task_name, data))

        # Run MJX batch evaluation
        results = evaluate_batch_mjx(model, params, task_data_list, config, mj_model, verbose=args.verbose,
                                     dump_dir=args.dump_rollout, norm_stats=norm_stats)

    # Compute average across all tasks at each window size
    # Filter out CLASSIFICATION* and AVERAGE keys (only include task results)
    task_results = {k: v for k, v in results.items() if not k.startswith('CLASSIFICATION') and k != 'AVERAGE'}

    if task_results:
        avg_results = {'n_steps': 1000}

        # Average for each window size
        for window in window_sizes:
            key_j1 = f'J1@{window}'
            if key_j1 in list(task_results.values())[0]:
                avg_results[f'J1@{window}'] = np.mean([r[f'J1@{window}'] for r in task_results.values() if f'J1@{window}' in r])
                avg_results[f'J2@{window}'] = np.mean([r[f'J2@{window}'] for r in task_results.values() if f'J2@{window}' in r])
                avg_results[f'J3@{window}'] = np.mean([r[f'J3@{window}'] for r in task_results.values() if f'J3@{window}' in r])
                avg_results[f'J4@{window}'] = np.mean([r[f'J4@{window}'] for r in task_results.values() if f'J4@{window}' in r])
                avg_results[f'J5@{window}'] = np.mean([r[f'J5@{window}'] for r in task_results.values() if f'J5@{window}' in r])
                avg_results[f'J6@{window}'] = np.mean([r[f'J6@{window}'] for r in task_results.values() if f'J6@{window}' in r])

        # Average for full trajectory
        avg_results['J1'] = np.mean([r['J1'] for r in task_results.values()])
        avg_results['J2'] = np.mean([r['J2'] for r in task_results.values()])
        avg_results['J3'] = np.mean([r['J3'] for r in task_results.values()])
        avg_results['J4'] = np.mean([r['J4'] for r in task_results.values()])
        avg_results['J5'] = np.mean([r['J5'] for r in task_results.values()])
        avg_results['J6'] = np.mean([r['J6'] for r in task_results.values()])

        results['AVERAGE'] = avg_results

        print("\n" + "=" * 80)
        print("AVERAGE MAE across all tasks (by window size):")
        print("=" * 80)
        print(f"{'Window':<10} {'J1 (deg)':<10} {'J2 (deg)':<10} {'J3 (deg)':<10} {'J4 (deg)':<10} {'J5 (mm)':<10}")
        print("-" * 70)
        for window in window_sizes:
            key = f'J1@{window}'
            if key in avg_results:
                print(f"{window:<10} {avg_results[f'J1@{window}']:<10.2f} {avg_results[f'J2@{window}']:<10.2f} "
                      f"{avg_results[f'J3@{window}']:<10.2f} {avg_results[f'J4@{window}']:<10.2f} {avg_results[f'J5@{window}']:<10.2f}")
        print("-" * 70)
        print(f"{'Full':<10} {avg_results['J1']:<10.2f} {avg_results['J2']:<10.2f} "
              f"{avg_results['J3']:<10.2f} {avg_results['J4']:<10.2f} {avg_results['J5']:<10.2f}")

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # Generate LaTeX table
    data_dt = config['data_dt']
    generate_latex_table(results, args.output.replace('.json', '_table.tex'), data_dt=data_dt)


def generate_latex_table(results, output_path, data_dt=0.017):
    """Generate LaTeX table from results with different prediction horizons."""
    # Window sizes and corresponding times
    window_sizes = [10, 100, 200, 300, 400, 500, 600]

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
    latex.append(r"  \caption{Simulation accuracy across different time horizons. Values show mean absolute error on the test set.}")
    latex.append(r"  \label{tab:sim-accuracy}")
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

    # Data rows for J1-J4 (revolute)
    for j in range(1, 5):
        joint_name = f"Joint{j}"
        values = [f"{avg.get(f'J{j}@{w}', 0):.2f}" for w in available_windows]
        latex.append(f"    {joint_name} & Revolute & deg & " + " & ".join(values) + r" \\")

    latex.append(r"    \midrule")

    # Data rows for J5-J6 (prismatic)
    for j in range(5, 7):
        joint_name = f"Joint{j}"
        values = [f"{avg.get(f'J{j}@{w}', 0):.2f}" for w in available_windows]
        latex.append(f"    {joint_name} & Prismatic & mm & " + " & ".join(values) + r" \\")

    latex.append(r"    \bottomrule")
    latex.append(r"  \end{tabular}%")
    latex.append(r"  }")
    latex.append(r"\end{table}")

    with open(output_path, 'w') as f:
        f.write('\n'.join(latex))
    print(f"LaTeX table saved to {output_path}")


if __name__ == "__main__":
    main()
