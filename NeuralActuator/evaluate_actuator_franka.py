"""
Evaluate Neural Actuator (Franka Panda version) on test sets.

Computes per-joint MAE for each task and outputs results in JSON format.
Joint 1-7: revolute arm joints (degrees). J8 is the parallel gripper (mm);
the gripper is position-controlled on the real robot, so the simulation
replays it from the recorded width instead of torque control.

Rollout dumps (--dump_rollout) use the same npz keys as the other platforms
(sim_q, gt_q, force_pred, force_gt, gate, cond). sim_q/gt_q have 8 columns:
7 arm joints in radians plus the finger joint position in meters ([0, 0.04]).
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
N_JOINTS = 7
FEATURE_DIM = 52
# Panda finger joint travel in meters; gripper_width in the CSVs is normalized
# to [0, 1], so finger qpos = gripper_width * FINGER_TRAVEL.
FINGER_TRAVEL = 0.04


def load_model(model_path, config):
    """Load trained model from pickle file.

    Returns (model, params, history_length, feature_dim). Checkpoints written
    with EMA tracking store {'params', 'ema_params'}; set use_ema_params in the
    config to evaluate the EMA weights.
    """
    with open(model_path, 'rb') as f:
        params = pickle.load(f)

    if isinstance(params, dict) and 'ema_params' in params:
        if config.get('use_ema_params', False):
            print("  Using EMA parameters from checkpoint")
            params = params['ema_params']
        else:
            params = params['params']

    model_type = get_model_type_from_config(config)
    hidden_dim = config['hidden_dim']
    latent_dim = config['latent_dim']
    history_length = config['history_length']
    dropout_rate = config['dropout_rate']

    # Franka version: 52D features (7 arm joints + gripper channels)
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

    return model, params, history_length, feature_dim


def load_csv_data(csv_path, cfg=None):
    """Load CSV data and extract relevant columns for the Franka."""
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    # Trim start/end of trajectory (remove noisy boundaries; must match training)
    trim_start = float(cfg.get('trim_start', 0.0)) if cfg else 0.0
    trim_end = float(cfg.get('trim_end', 0.0)) if cfg else 0.0
    if trim_start > 0 or trim_end > 0:
        n = len(df)
        df = df.iloc[int(n * trim_start):int(n * (1 - trim_end))].reset_index(drop=True)

    # Joint positions (pos1-7)
    pos_cols = [f'pos{i}' for i in range(1, 8)]
    positions = df[pos_cols].values  # (N, 7)

    # Target (commanded) pose channel. The model is command-conditioned: at run
    # time an external source (controller, teleoperation, or IK) supplies this
    # target before the actuator acts. The recorded cmd_pos holds only sparse
    # waypoint setpoints, so offline we proxy the command with a short-horizon
    # reference off the achieved trajectory, pos[t+K] * scale. Stiff position
    # tracking makes the near-future achieved pose a good stand-in for the command;
    # the scale compensates for the small command-to-achieved tracking lag.
    lookahead_frames = int(cfg.get('lookahead_frames', 5)) if cfg else 5
    lookahead_scale = float(cfg.get('lookahead_scale', 1.03)) if cfg else 1.03
    n_rows = len(df)
    cmd_positions = np.zeros((n_rows, 7), dtype=np.float32)
    for t in range(n_rows):
        future_t = min(t + lookahead_frames, n_rows - 1)
        cmd_positions[t] = positions[future_t] * lookahead_scale

    # Gripper width (normalized 0-1)
    gripper_width = df['gripper_width'].values  # (N,)

    # Commanded torque (tau_d1-7, Nm)
    tau_d = df[[f'tau_d{i}' for i in range(1, 8)]].values  # (N, 7)

    # Joint velocities (rad/s)
    velocities = df[[f'vel{i}' for i in range(1, 8)]].values  # (N, 7)

    # Motor-side positions and velocities
    motor_pos = df[[f'motor_pos{i}' for i in range(1, 8)]].values  # (N, 7)
    motor_vel = df[[f'motor_vel{i}' for i in range(1, 8)]].values  # (N, 7)

    # Timestamps
    timestamps = df['timestamp'].values if 'timestamp' in df.columns else np.arange(len(df)) * 0.016

    # Force GT: force_x/y are -999 sentinels (no reading), force_z is the
    # payload weight (-mg). Sentinels map to 0 N as on the other platforms.
    force_cols = ['force_x', 'force_y', 'force_z']
    if all(c in df.columns for c in force_cols):
        forces = df[force_cols].values.copy().astype(np.float32)  # (N, 3)
        forces[forces == -999] = 0.0
        has_force = bool(np.any(np.abs(forces) > 0.01))
    else:
        forces = np.zeros((len(df), 3), dtype=np.float32)
        has_force = False

    return {
        'positions': positions,           # (N, 7)
        'cmd_positions': cmd_positions,   # (N, 7) lookahead targets
        'gripper_width': gripper_width,   # (N,)
        'tau_d': tau_d,                   # (N, 7)
        'velocities': velocities,         # (N, 7)
        'motor_pos': motor_pos,           # (N, 7)
        'motor_vel': motor_vel,           # (N, 7)
        'timestamps': timestamps,
        'forces': forces,
        'has_force': has_force,
        'n_samples': len(df)
    }


def build_features(cmd_pos, pos, gripper, tau_d, vel, motor_pos, motor_vel):
    """Build 52D feature vector for the Franka.

    52D Feature Vector:
    - 0-6:   lookahead_pos1-7 (pos[t+K] * scale, rad)
    - 7-13:  pos1-7 (rad)
    - 14:    gripper_width (normalized 0-1)
    - 15-21: tau_d1-7 (commanded torque, Nm)
    - 22-28: vel1-7 (rad/s)
    - 29-35: motor_pos1-7 (rad)
    - 36-42: motor_vel1-7 (rad/s)
    - 43:    goal_gripper (= gripper_width, no independent goal)
    - 44-50: arm_error1-7 = lookahead_pos - pos (rad)
    - 51:    gripper_error = goal_gripper - gripper_width (= 0 from CSV)
    """
    arm_error = cmd_pos - pos
    gripper_error = 0.0  # goal_gripper equals gripper_width in the data

    features = np.concatenate([
        cmd_pos,           # 0-6
        pos,               # 7-13
        [gripper],         # 14
        tau_d,             # 15-21
        vel,               # 22-28
        motor_pos,         # 29-35
        motor_vel,         # 36-42
        [gripper],         # 43
        arm_error,         # 44-50
        [gripper_error],   # 51
    ])
    return features


def build_features_jax(cmd_pos, pos, gripper, tau_d, vel, motor_pos, motor_vel):
    """Build 52D feature vector for the Franka (JAX version)."""
    arm_error = cmd_pos - pos
    gripper_error = jnp.float32(0.0)

    features = jnp.concatenate([
        cmd_pos,                       # 0-6
        pos,                           # 7-13
        jnp.array([gripper]),          # 14
        tau_d,                         # 15-21
        vel,                           # 22-28
        motor_pos,                     # 29-35
        motor_vel,                     # 36-42
        jnp.array([gripper]),          # 43
        arm_error,                     # 44-50
        jnp.array([gripper_error]),    # 51
    ])
    return features


def _get_cache_key(config, mj_model, max_len):
    """Generate cache key based on config, model, and data dimensions."""
    EVAL_LOGIC_VERSION = "franka_v1"
    key_parts = [
        EVAL_LOGIC_VERSION,  # Invalidates cache when eval logic changes
        str(config['history_length']),
        str(config['data_dt']),
        str(config['sim_step_size']),
        str(mj_model.nq),
        str(mj_model.nv),
        str(mj_model.nu),
        str(max_len),  # Include max_len since it affects JIT compilation
    ]
    return hashlib.md5('|'.join(key_parts).encode()).hexdigest()


def _build_eval_function(model, mjx_model, mjx_data_single, config, max_len):
    """Build JIT-compiled evaluation function with params as explicit argument."""
    history_length = config['history_length']
    data_dt = config['data_dt']
    sim_step_size = config['sim_step_size']
    feature_dim = config.get('feature_dim', FEATURE_DIM)
    # Residual torque mode
    use_residual_torque = bool(config.get('use_residual_torque', False))
    torque_constant = float(config.get('torque_constant', 1.0))

    def single_step(carry, inputs):
        """Single simulation step for one task."""
        qpos, qvel, history, params = carry
        csv_feat, step_idx, traj_len = inputs

        # Valid when step_idx < traj_len - 1 (need next step for GT)
        valid = step_idx < (traj_len - 1)

        sim_pos_arm = qpos[:N_JOINTS]
        sim_vel = qvel[:N_JOINTS]
        gripper_normalized = qpos[7] / FINGER_TRAVEL

        current_feat = csv_feat
        # Update pos (7-13) with sim
        current_feat = current_feat.at[7:14].set(sim_pos_arm)
        # Update gripper (14) with sim (normalized)
        current_feat = current_feat.at[14].set(gripper_normalized)
        # Update vel (22-28) with sim
        current_feat = current_feat.at[22:29].set(sim_vel)
        # Update arm_error (44-50)
        arm_error = current_feat[0:7] - sim_pos_arm
        current_feat = current_feat.at[44:51].set(arm_error)
        # Update gripper_error (51)
        gripper_error = current_feat[43] - gripper_normalized
        current_feat = current_feat.at[51].set(gripper_error)

        hist_flat = history.reshape(-1)

        pred_tau, final_force, raw_force, gate, condition_pred, _ = model.apply(
            params, hist_flat[None, :], current_feat[None, :],
            None, ts=data_dt, training=False
        )

        # Residual torque mode: final_torque = base_torque + network_output
        # tau_d is at indices 15-21 (Nm already)
        if use_residual_torque:
            tau_d_values = csv_feat[15:22]
            base_torque = tau_d_values * torque_constant
            tau = base_torque + pred_tau[0]
        else:
            tau = pred_tau[0]

        mjx_d = mjx_data_single.replace(
            qpos=qpos,
            qvel=qvel,
            ctrl=jnp.zeros(mjx_model.nu).at[:N_JOINTS].set(tau)
        )

        def sim_body(i, d):
            d_new = mjx.step(mjx_model, d)
            d_new = d_new.replace(
                qpos=d_new.qpos.astype(jnp.float32),
                qvel=d_new.qvel.astype(jnp.float32)
            )
            # The Panda gripper is a tendon-coupled actuator; with x64 enabled
            # MJX promotes its int32 wrap bookkeeping to int64 inside step(),
            # which then fails to unify with the carry. Cast back explicitly.
            d_new = d_new.tree_replace({
                '_impl.ten_wrapadr': d_new._impl.ten_wrapadr.astype(jnp.int32),
                '_impl.ten_wrapnum': d_new._impl.ten_wrapnum.astype(jnp.int32),
                '_impl.wrap_obj': d_new._impl.wrap_obj.astype(jnp.int32),
            })
            return d_new

        mjx_d = jax.lax.fori_loop(0, sim_step_size, sim_body, mjx_d)

        # Gripper: replay from the recorded width (position-controlled on the
        # real robot, so it is not part of the torque rollout)
        gripper_gt_m = csv_feat[14] * FINGER_TRAVEL
        new_qpos = mjx_d.qpos.at[7].set(gripper_gt_m)
        new_qpos = new_qpos.at[8].set(gripper_gt_m)
        new_qpos = new_qpos.astype(jnp.float32)
        # qvel: NaN/Inf protection + clamp (must match training)
        new_qvel = jnp.nan_to_num(mjx_d.qvel, nan=0.0, posinf=0.0, neginf=0.0)
        new_qvel = jnp.clip(new_qvel, -100.0, 100.0).astype(jnp.float32)

        new_history = jnp.roll(history, -1, axis=0)
        new_history = new_history.at[-1].set(current_feat)

        final_qpos = jnp.where(valid, new_qpos, qpos)
        final_qvel = jnp.where(valid, new_qvel, qvel)
        final_history = jnp.where(valid, new_history, history)

        # 8D: 7 arm joints (rad) + finger joint (m)
        sim_q = jnp.concatenate([final_qpos[:N_JOINTS], final_qpos[7:8]])
        cond_vals = condition_pred[0]  # (7,)
        force_pred = final_force[0]    # (3,) - gated force
        gate_val = gate[0, 0]          # scalar

        return (final_qpos, final_qvel, final_history, params), (sim_q, cond_vals, force_pred, gate_val)

    def eval_single_task(params, init_pos, init_history, csv_feats, traj_len):
        """Evaluate single task using scan.

        Starts from index 0 with zero-padded history (aligned with training).
        Returns sim positions and condition/force/gate predictions.
        """
        qpos = jnp.zeros(mjx_model.nq, dtype=jnp.float32)
        qpos = qpos.at[:N_JOINTS].set(init_pos[:N_JOINTS])
        qpos = qpos.at[7].set(init_pos[7])   # finger_joint1
        qpos = qpos.at[8].set(init_pos[7])   # finger_joint2
        qvel = jnp.zeros(mjx_model.nv, dtype=jnp.float32)

        n_eval_steps = max_len - 1
        step_indices = jnp.arange(n_eval_steps, dtype=jnp.int32)

        csv_feats_eval = csv_feats[0:n_eval_steps]

        init_carry = (qpos, qvel, init_history, params)
        inputs = (csv_feats_eval, step_indices, jnp.broadcast_to(traj_len, (n_eval_steps,)))

        _, (recorded_sim_q, recorded_cond, recorded_force, recorded_gate) = jax.lax.scan(single_step, init_carry, inputs)

        return recorded_sim_q, recorded_cond, recorded_force, recorded_gate

    return jax.jit(eval_single_task)


def evaluate_batch_mjx(model, params, task_data_list, config, mj_model, verbose=True, dump_dir=None):
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
        dump_dir: Optional directory for per-task rollout npz dumps

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
    cache_key = _get_cache_key(config, mj_model, max_len)
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
        eval_single_task_jit = _build_eval_function(model, mjx_model, mjx_data_single, config, max_len)
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
    all_target_pos = []    # (n_tasks, max_len, 8) - 7 arm + 1 gripper
    all_target_force = []  # (n_tasks, max_len, 3)
    all_init_pos = []      # (n_tasks, 8)
    all_lengths = []       # (n_tasks,)
    all_has_force = []     # (n_tasks,) - whether each task has force data

    for task_name, data in task_data_list:
        n_samples = data['n_samples']
        all_lengths.append(n_samples)
        all_has_force.append(data.get('has_force', False))

        # Build CSV features for entire trajectory
        csv_feats = np.zeros((max_len, feature_dim), dtype=np.float32)
        target_pos = np.zeros((max_len, 8), dtype=np.float32)
        target_force = np.zeros((max_len, 3), dtype=np.float32)

        for i in range(n_samples):
            csv_feats[i] = build_features(
                cmd_pos=data['cmd_positions'][i],
                pos=data['positions'][i],
                gripper=data['gripper_width'][i],
                tau_d=data['tau_d'][i],
                vel=data['velocities'][i],
                motor_pos=data['motor_pos'][i],
                motor_vel=data['motor_vel'][i],
            )

            if i < n_samples - 1:
                # Target is next position
                target_pos[i, :N_JOINTS] = data['positions'][i + 1]
                target_pos[i, 7] = data['gripper_width'][i + 1] * FINGER_TRAVEL
                # Target force (current step force as GT)
                target_force[i] = data['forces'][i]

        all_csv_features.append(csv_feats)
        all_target_pos.append(target_pos)
        all_target_force.append(target_force)

        # Initial position (at index 0, aligned with training zero-padding)
        init_pos = np.zeros(8, dtype=np.float32)
        init_pos[:N_JOINTS] = data['positions'][0]
        init_pos[7] = data['gripper_width'][0] * FINGER_TRAVEL
        all_init_pos.append(init_pos)

        # Debug: verify first task's init position
        if len(all_init_pos) == 1 and verbose:
            print(f"  [DEBUG] First task init pos (index 0): {np.rad2deg(init_pos[:N_JOINTS])}")

    # Convert to JAX arrays
    if verbose:
        print(f"\n[Step 3] Converting to JAX arrays...")
        t0 = time.time()
    csv_features_jax = jnp.array(np.stack(all_csv_features))  # (n_tasks, max_len, feature_dim)
    target_pos_jax = jnp.array(np.stack(all_target_pos))      # (n_tasks, max_len, 8)
    target_force_jax = jnp.array(np.stack(all_target_force))  # (n_tasks, max_len, 3)
    init_pos_jax = jnp.array(np.stack(all_init_pos))          # (n_tasks, 8)
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
    window_sizes = [10, 100, 200, 300, 400, 500, 600]

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

        # Build GT positions (starting from index 0): 7 arm joints + finger (m)
        recorded_gt_q = np.column_stack([
            data['positions'][1:actual_len + 1],
            data['gripper_width'][1:actual_len + 1] * FINGER_TRAVEL,
        ]).astype(np.float32)

        # Optionally dump rollout trajectories (for rendering / analysis).
        # sim_q/gt_q are 8 columns: 7 arm joints (rad) + finger joint (m).
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

            # 7 arm joints in degrees
            joint_errors_deg = np.rad2deg(sim_q_window[:, :N_JOINTS] - gt_q_window[:, :N_JOINTS])
            mae_joints = np.mean(np.abs(joint_errors_deg), axis=0)

            # Gripper in mm
            gripper_error_mm = (sim_q_window[:, 7] - gt_q_window[:, 7]) * 1000.0
            mae_grip = np.mean(np.abs(gripper_error_mm))

            for j in range(N_JOINTS):
                results[f'J{j+1}@{window}'] = float(mae_joints[j])
            results[f'J8@{window}'] = float(mae_grip)  # Gripper as J8

            # Force MAE at this window (if has force data)
            if has_force_data:
                force_error = recorded_force[:window] - target_force_task[:window]
                results[f'Force@{window}'] = float(np.mean(np.abs(force_error)))
                results[f'ForceZ@{window}'] = float(np.mean(np.abs(force_error[:, 2])))

        # Full trajectory MAE
        joint_errors_deg = np.rad2deg(recorded_sim_q[:, :N_JOINTS] - recorded_gt_q[:, :N_JOINTS])
        mae_joints = np.mean(np.abs(joint_errors_deg), axis=0)
        gripper_error_mm = (recorded_sim_q[:, 7] - recorded_gt_q[:, 7]) * 1000.0
        mae_grip = np.mean(np.abs(gripper_error_mm))

        for j in range(N_JOINTS):
            results[f'J{j+1}'] = float(mae_joints[j])
        results['J8'] = float(mae_grip)

        # Force MAE (full trajectory)
        if has_force_data:
            force_error = recorded_force - target_force_task
            results['Force'] = float(np.mean(np.abs(force_error)))
            results['ForceZ'] = float(np.mean(np.abs(force_error[:, 2])))
            results['has_force'] = True
        else:
            results['has_force'] = False

        all_results[task_name] = results

        task_time = time.time() - task_start
        force_str = f" Force={results.get('Force', 0):.2f}N" if has_force_data else ""
        joint_str = " ".join([f"J{j+1}={results[f'J{j+1}']:.1f}°" for j in range(N_JOINTS)])
        print(f"@{actual_len}: {joint_str} Grip={results['J8']:.1f}mm{force_str} ({task_time:.2f}s)")

    if verbose:
        total_time = time.time() - total_start
        eval_time = time.time() - eval_start
        print(f"\n[Summary]")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Evaluation time: {eval_time:.2f}s")
        print(f"  Avg per task: {eval_time / n_tasks:.2f}s")

    return all_results


def main():
    parser = argparse.ArgumentParser(description='Evaluate Neural Actuator (Franka) on test sets')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--output', type=str, default='outputs/evaluation_results_franka.json', help='Output JSON file')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--dump_rollout', type=str, default=None, help='Directory to save per-task rollout trajectories (npz)')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Load model
    print(f"Loading model from {args.model_path}...")
    model, params, history_length, feature_dim = load_model(args.model_path, config)

    # Load MuJoCo model (chdir so the MJCF finds its mesh assets)
    mjcf_path = config.get('mjcf_path', 'robot_franka/scene.xml')
    print(f"Loading MuJoCo model from {mjcf_path}...")
    abs_mjcf_path = os.path.abspath(mjcf_path)
    model_dir = os.path.dirname(abs_mjcf_path)
    cwd = os.getcwd()
    try:
        os.chdir(model_dir)
        mj_model = mujoco.MjModel.from_xml_path(os.path.basename(abs_mjcf_path))
    finally:
        os.chdir(cwd)

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

    # Load all task data
    print("\nLoading task data...")
    task_data_list = []
    for task_name, csv_path in valid_tasks:
        data = load_csv_data(csv_path, config)
        task_data_list.append((task_name, data))

    # Run MJX batch evaluation
    results = evaluate_batch_mjx(model, params, task_data_list, config, mj_model, verbose=args.verbose,
                                 dump_dir=args.dump_rollout)

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
                avg_results[f'J8@{window}'] = np.mean(
                    [r[f'J8@{window}'] for r in task_results.values() if f'J8@{window}' in r])
                force_vals = [r[f'Force@{window}'] for r in task_results.values() if f'Force@{window}' in r]
                if force_vals:
                    avg_results[f'Force@{window}'] = np.mean(force_vals)

        # Average for full trajectory
        for j in range(N_JOINTS):
            avg_results[f'J{j+1}'] = np.mean([r[f'J{j+1}'] for r in task_results.values()])
        avg_results['J8'] = np.mean([r['J8'] for r in task_results.values()])
        force_vals = [r['Force'] for r in task_results.values() if r.get('has_force', False)]
        if force_vals:
            avg_results['Force'] = np.mean(force_vals)

        results['AVERAGE'] = avg_results

        print("\n" + "=" * 110)
        print("AVERAGE MAE across all tasks (by window size):")
        print("=" * 110)
        header = f"{'Window':<10} " + " ".join([f"{'J' + str(j+1) + ' (deg)':<10}" for j in range(N_JOINTS)])
        header += f"{'Grip (mm)':<10} {'Force (N)':<10}"
        print(header)
        print("-" * 110)
        for window in window_sizes:
            key = f'J1@{window}'
            if key in avg_results:
                row = f"{window:<10} " + " ".join([f"{avg_results[f'J{j+1}@{window}']:<10.2f}" for j in range(N_JOINTS)])
                row += f"{avg_results[f'J8@{window}']:<10.2f} "
                row += f"{avg_results.get(f'Force@{window}', 0.0):<10.2f}"
                print(row)
        print("-" * 110)
        row = f"{'Full':<10} " + " ".join([f"{avg_results[f'J{j+1}']:<10.2f}" for j in range(N_JOINTS)])
        row += f"{avg_results['J8']:<10.2f} "
        row += f"{avg_results.get('Force', 0.0):<10.2f}"
        print(row)

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # Generate LaTeX table
    generate_latex_table(results, args.output.replace('.json', '_table.tex'), data_dt=config['data_dt'])


def generate_latex_table(results, output_path, data_dt=0.016):
    """Generate LaTeX table from results with different prediction horizons."""
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
    latex.append(r"  \caption{Simulation accuracy across different time horizons on the Franka Panda. Values show mean absolute error on the test set.}")
    latex.append(r"  \label{tab:sim-accuracy-franka}")
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

    # Data rows for J1-J7 (revolute)
    for j in range(1, N_JOINTS + 1):
        joint_name = f"Joint{j}"
        values = [f"{avg.get(f'J{j}@{w}', 0):.2f}" for w in available_windows]
        latex.append(f"    {joint_name} & Revolute & deg & " + " & ".join(values) + r" \\")

    latex.append(r"    \midrule")

    # Data row for the gripper
    values = [f"{avg.get(f'J8@{w}', 0):.2f}" for w in available_windows]
    latex.append(f"    Gripper & Prismatic & mm & " + " & ".join(values) + r" \\")

    latex.append(r"    \bottomrule")
    latex.append(r"  \end{tabular}%")
    latex.append(r"  }")
    latex.append(r"\end{table}")

    with open(output_path, 'w') as f:
        f.write('\n'.join(latex))
    print(f"LaTeX table saved to {output_path}")


if __name__ == "__main__":
    main()
