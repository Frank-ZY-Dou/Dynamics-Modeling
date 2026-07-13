"""
Run a trained Neural Actuator checkpoint on a single trajectory CSV.

Default mode rolls out the model in MJX from the trajectory start (same protocol
as the eval scripts) and writes per-step predictions (simulated joint positions,
torque, force, gate) to an npz or csv file, printing per-joint MAE against
the CSV ground truth. Force MAE is reported when the CSV carries force labels.
Supports all platforms via --robot omx|so101|franka.

--force_only skips the simulator entirely: the feature history is built from the
CSV telemetry rows themselves (as it would be from a live robot stream) and the
network runs one forward pass per step, outputting force, gate and torque. This
is the deployment path for force perception on hardware; no MuJoCo model is
loaded and the output carries no sim_q. Since there is no rollout, per-joint MAE
is skipped and only force MAE is reported.
"""

import argparse
import os
import time

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np
import pandas as pd
import yaml


def _build_rollout_omx(model, mjx_model, mjx_data_single, config, norm_stats):
    """JIT rollout for the OpenManipulator-X (mirrors evaluate_actuator, also records torque)."""
    data_dt = config['data_dt']
    sim_step_size = config['sim_step_size']
    use_residual_torque = config['use_residual_torque']
    torque_constant = float(config['torque_constant'])
    gripper_torque_clip = config.get('gripper_torque_clip', None)
    qvel_clip = float(config.get('qvel_clip', 0.0))

    if norm_stats is not None:
        norm_mean = jnp.array(np.asarray(norm_stats[0]))
        norm_std = jnp.array(np.asarray(norm_stats[1]))

    def normalize_feat(x):
        if norm_stats is None:
            return x
        return jnp.clip((x - norm_mean) / norm_std, -10.0, 10.0)

    def single_step(carry, csv_feat):
        qpos, qvel, history, params = carry

        sim_pos = qpos[:5]
        sim_vel = qvel[:5]
        aperture_val = qpos[4] * 1000.0  # m to mm

        current_feat = csv_feat
        current_feat = current_feat.at[5:9].set(sim_pos[:4])
        current_feat = current_feat.at[9].set(aperture_val)
        current_feat = current_feat.at[15:20].set(sim_vel)
        arm_error = current_feat[0:4] - sim_pos[:4]
        current_feat = current_feat.at[31:35].set(arm_error)
        gripper_error = current_feat[30] - aperture_val
        current_feat = current_feat.at[35].set(gripper_error)

        hist_flat = history.reshape(-1)
        net_feat = normalize_feat(current_feat)
        pred_tau, final_force, raw_force, gate, condition_pred, _ = model.apply(
            params, hist_flat[None, :], net_feat[None, :],
            None, ts=data_dt, training=False
        )

        if use_residual_torque:
            current_values = csv_feat[10:14]  # current1-4 in mA (arm only)
            base_torque = (current_values / 1000.0) * torque_constant
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
            return d_new.replace(
                qpos=d_new.qpos.astype(jnp.float32),
                qvel=d_new.qvel.astype(jnp.float32)
            )

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

        outputs = (new_qpos[:5], tau, final_force[0], gate[0, 0], condition_pred[0])
        return (new_qpos, new_qvel, new_history, params), outputs

    def rollout(params, init_pos, init_history, csv_feats):
        qpos = jnp.zeros(mjx_model.nq, dtype=jnp.float32)
        qpos = qpos.at[:4].set(init_pos[:4])
        qpos = qpos.at[4].set(init_pos[4])
        qpos = qpos.at[5].set(init_pos[4])
        qvel = jnp.zeros(mjx_model.nv, dtype=jnp.float32)

        init_carry = (qpos, qvel, init_history, params)
        _, outputs = jax.lax.scan(single_step, init_carry, csv_feats)
        return outputs

    return jax.jit(rollout)


def _build_rollout_so101(model, mjx_model, mjx_data_single, config, norm_stats, n_joints, vel_counts_per_rad):
    """JIT rollout for the SO-101 (mirrors evaluate_actuator_so101, also records torque)."""
    data_dt = config['data_dt']
    sim_step_size = config['sim_step_size']
    torque_clip = float(config.get('torque_clip', 3.0))
    qvel_clip = float(config.get('qvel_clip', 0.0))

    if norm_stats is not None:
        norm_mean = jnp.array(np.asarray(norm_stats[0]))
        norm_std = jnp.array(np.asarray(norm_stats[1]))

    def normalize_feat(x):
        if norm_stats is None:
            return x
        return jnp.clip((x - norm_mean) / norm_std, -10.0, 10.0)

    def single_step(carry, csv_feat):
        qpos, qvel, history, params = carry

        sim_pos = qpos[:n_joints]
        sim_vel = qvel[:n_joints]

        current_feat = csv_feat
        current_feat = current_feat.at[6:12].set(sim_pos)
        current_feat = current_feat.at[18:24].set(sim_vel * vel_counts_per_rad)
        pos_error = current_feat[0:6] - sim_pos
        current_feat = current_feat.at[36:42].set(pos_error)

        hist_flat = history.reshape(-1)
        net_feat = normalize_feat(current_feat)
        pred_tau, final_force, raw_force, gate, condition_pred, _ = model.apply(
            params, hist_flat[None, :], net_feat[None, :],
            None, ts=data_dt, training=False
        )

        tau = jnp.clip(pred_tau[0], -torque_clip, torque_clip)

        mjx_d = mjx_data_single.replace(
            qpos=qpos,
            qvel=qvel,
            ctrl=tau.astype(jnp.float32)
        )

        def sim_body(i, d):
            d_new = mjx.step(mjx_model, d)
            return d_new.replace(
                qpos=d_new.qpos.astype(jnp.float32),
                qvel=d_new.qvel.astype(jnp.float32)
            )

        mjx_d = jax.lax.fori_loop(0, sim_step_size, sim_body, mjx_d)

        new_qpos = mjx_d.qpos.astype(jnp.float32)
        new_qvel = mjx_d.qvel.astype(jnp.float32)
        if qvel_clip > 0:
            new_qvel = jnp.nan_to_num(jnp.clip(new_qvel, -qvel_clip, qvel_clip),
                                      nan=0.0, posinf=qvel_clip, neginf=-qvel_clip)

        new_history = jnp.roll(history, -1, axis=0)
        new_history = new_history.at[-1].set(net_feat)

        outputs = (new_qpos[:n_joints], tau, final_force[0], gate[0, 0], condition_pred[0])
        return (new_qpos, new_qvel, new_history, params), outputs

    def rollout(params, init_pos, init_history, csv_feats):
        qpos = jnp.zeros(mjx_model.nq, dtype=jnp.float32)
        qpos = qpos.at[:n_joints].set(init_pos)
        qvel = jnp.zeros(mjx_model.nv, dtype=jnp.float32)

        init_carry = (qpos, qvel, init_history, params)
        _, outputs = jax.lax.scan(single_step, init_carry, csv_feats)
        return outputs

    return jax.jit(rollout)


def _build_rollout_franka(model, mjx_model, mjx_data_single, config, n_joints, finger_travel):
    """JIT rollout for the Franka Panda (mirrors evaluate_actuator_franka, also records torque).

    The gripper is position-controlled on the real robot, so its finger joints
    replay the recorded width instead of being driven by predicted torque.
    Features are fed to the network unnormalized, matching training.
    """
    data_dt = config['data_dt']
    sim_step_size = config['sim_step_size']
    use_residual_torque = bool(config.get('use_residual_torque', False))
    torque_constant = float(config.get('torque_constant', 1.0))

    def single_step(carry, csv_feat):
        qpos, qvel, history, params = carry

        sim_pos = qpos[:n_joints]
        sim_vel = qvel[:n_joints]
        gripper_normalized = qpos[7] / finger_travel

        current_feat = csv_feat
        current_feat = current_feat.at[7:14].set(sim_pos)
        current_feat = current_feat.at[14].set(gripper_normalized)
        current_feat = current_feat.at[22:29].set(sim_vel)
        arm_error = current_feat[0:7] - sim_pos
        current_feat = current_feat.at[44:51].set(arm_error)
        gripper_error = current_feat[43] - gripper_normalized
        current_feat = current_feat.at[51].set(gripper_error)

        hist_flat = history.reshape(-1)
        pred_tau, final_force, raw_force, gate, condition_pred, _ = model.apply(
            params, hist_flat[None, :], current_feat[None, :],
            None, ts=data_dt, training=False
        )

        if use_residual_torque:
            base_torque = csv_feat[15:22] * torque_constant  # tau_d1-7 in Nm
            tau = base_torque + pred_tau[0]
        else:
            tau = pred_tau[0]

        mjx_d = mjx_data_single.replace(
            qpos=qpos,
            qvel=qvel,
            ctrl=jnp.zeros(mjx_model.nu, dtype=jnp.float32).at[:n_joints].set(tau)
        )

        def sim_body(i, d):
            d_new = mjx.step(mjx_model, d)
            d_new = d_new.replace(
                qpos=d_new.qpos.astype(jnp.float32),
                qvel=d_new.qvel.astype(jnp.float32)
            )
            # The Panda gripper is a tendon-coupled actuator; cast its wrap
            # bookkeeping back to int32 so the dtypes unify with the scan carry.
            d_new = d_new.tree_replace({
                '_impl.ten_wrapadr': d_new._impl.ten_wrapadr.astype(jnp.int32),
                '_impl.ten_wrapnum': d_new._impl.ten_wrapnum.astype(jnp.int32),
                '_impl.wrap_obj': d_new._impl.wrap_obj.astype(jnp.int32),
            })
            return d_new

        mjx_d = jax.lax.fori_loop(0, sim_step_size, sim_body, mjx_d)

        # Gripper: replay from the recorded width (position-controlled on the
        # real robot, so it is not part of the torque rollout)
        gripper_gt_m = csv_feat[14] * finger_travel
        new_qpos = mjx_d.qpos.at[7].set(gripper_gt_m)
        new_qpos = new_qpos.at[8].set(gripper_gt_m)
        new_qpos = new_qpos.astype(jnp.float32)
        # qvel: NaN/Inf protection + clamp (must match training)
        new_qvel = jnp.nan_to_num(mjx_d.qvel, nan=0.0, posinf=0.0, neginf=0.0)
        new_qvel = jnp.clip(new_qvel, -100.0, 100.0).astype(jnp.float32)

        new_history = jnp.roll(history, -1, axis=0)
        new_history = new_history.at[-1].set(current_feat)

        # 8 columns: 7 arm joints (rad) + finger joint (m)
        sim_q = jnp.concatenate([new_qpos[:n_joints], new_qpos[7:8]])
        outputs = (sim_q, tau, final_force[0], gate[0, 0], condition_pred[0])
        return (new_qpos, new_qvel, new_history, params), outputs

    def rollout(params, init_pos, init_history, csv_feats):
        qpos = jnp.zeros(mjx_model.nq, dtype=jnp.float32)
        qpos = qpos.at[:n_joints].set(init_pos[:n_joints])
        qpos = qpos.at[7].set(init_pos[7])   # finger_joint1
        qpos = qpos.at[8].set(init_pos[7])   # finger_joint2
        qvel = jnp.zeros(mjx_model.nv, dtype=jnp.float32)

        init_carry = (qpos, qvel, init_history, params)
        _, outputs = jax.lax.scan(single_step, init_carry, csv_feats)
        return outputs

    return jax.jit(rollout)


def _build_force_only(model, config, norm_stats, robot):
    """Teacher-forced forward pass: every feature comes from the CSV telemetry,
    no simulator in the loop. Returns (scan_fn, step_fn); step_fn is the
    single-step function a deployment loop would call."""
    data_dt = config['data_dt']

    if norm_stats is not None:
        norm_mean = jnp.array(np.asarray(norm_stats[0]))
        norm_std = jnp.array(np.asarray(norm_stats[1]))

    def normalize_feat(x):
        if norm_stats is None:
            return x
        return jnp.clip((x - norm_mean) / norm_std, -10.0, 10.0)

    # Torque post-processing, same as the rollout builders
    if robot == 'omx':
        use_residual_torque = config['use_residual_torque']
        torque_constant = float(config['torque_constant'])
        gripper_torque_clip = config.get('gripper_torque_clip', None)

        def postprocess_tau(pred_tau, csv_feat):
            if use_residual_torque:
                base_torque = (csv_feat[10:14] / 1000.0) * torque_constant  # current1-4 in mA
                tau = jnp.concatenate([base_torque + pred_tau[:4], pred_tau[4:5]])
            else:
                tau = pred_tau
            if gripper_torque_clip is not None:
                tau_limit = jnp.array([5.0, 5.0, 5.0, 5.0, float(gripper_torque_clip)])
                tau = jnp.clip(tau, -tau_limit, tau_limit)
            return tau
    elif robot == 'franka':
        use_residual_torque = bool(config.get('use_residual_torque', False))
        torque_constant = float(config.get('torque_constant', 1.0))

        def postprocess_tau(pred_tau, csv_feat):
            if use_residual_torque:
                base_torque = csv_feat[15:22] * torque_constant  # tau_d1-7 in Nm
                return base_torque + pred_tau
            return pred_tau
    else:
        torque_clip = float(config.get('torque_clip', 3.0))

        def postprocess_tau(pred_tau, csv_feat):
            return jnp.clip(pred_tau, -torque_clip, torque_clip)

    def single_step(carry, csv_feat):
        history, params = carry

        hist_flat = history.reshape(-1)
        net_feat = normalize_feat(csv_feat)
        pred_tau, final_force, raw_force, gate, condition_pred, _ = model.apply(
            params, hist_flat[None, :], net_feat[None, :],
            None, ts=data_dt, training=False
        )
        tau = postprocess_tau(pred_tau[0], csv_feat)

        new_history = jnp.roll(history, -1, axis=0)
        new_history = new_history.at[-1].set(net_feat)

        outputs = (tau, final_force[0], gate[0, 0], condition_pred[0])
        return (new_history, params), outputs

    def run(params, init_history, csv_feats):
        _, outputs = jax.lax.scan(single_step, (init_history, params), csv_feats)
        return outputs

    return jax.jit(run), jax.jit(single_step)


def main():
    parser = argparse.ArgumentParser(description='Roll out a trained Neural Actuator checkpoint on one trajectory CSV')
    parser.add_argument('--robot', type=str, default='omx', choices=['omx', 'so101', 'franka'],
                        help='Target platform')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to trained checkpoint (.pkl)')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--csv', type=str, required=True, help='Trajectory CSV to roll out')
    parser.add_argument('--out', type=str, default=None,
                        help='Output file (.npz or .csv); default outputs/<csv stem>_pred.npz')
    parser.add_argument('--use_ema', action='store_true', help='Use EMA weights from the checkpoint if present')
    parser.add_argument('--force_only', action='store_true',
                        help='No simulator: teacher-forced forward pass on the CSV telemetry, '
                             'outputs force/gate/torque per step (deployment mode)')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    config['use_ema_params'] = args.use_ema

    if args.robot == 'omx':
        import evaluate_actuator as ev
        n_joints = 5
        default_mjcf = 'robot/scene.xml'
    elif args.robot == 'franka':
        import evaluate_actuator_franka as ev
        n_joints = ev.N_JOINTS
        default_mjcf = 'robot_franka/scene.xml'
    else:
        import evaluate_actuator_so101 as ev
        n_joints = ev.N_JOINTS
        default_mjcf = 'robot_so101/so101_torque_scene.xml'

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    if args.robot == 'franka':
        # Franka checkpoints do not carry feature normalization stats
        model, params, history_length, feature_dim = ev.load_model(args.checkpoint, config)
        norm_stats = None
    else:
        model, params, history_length, feature_dim, norm_stats = ev.load_model(args.checkpoint, config)
    if norm_stats is not None:
        print("  Checkpoint carries feature normalization stats, applying at network input")

    # Load MuJoCo model with the same timestep and solver settings as training
    # (rollout mode only; --force_only never touches the simulator)
    if not args.force_only:
        mjcf_path = config.get('mjcf_path', default_mjcf)
        print(f"Loading MuJoCo model from {mjcf_path}...")
        if args.robot == 'franka':
            # chdir so the MJCF finds its mesh assets (same as the eval script)
            abs_mjcf_path = os.path.abspath(mjcf_path)
            cwd = os.getcwd()
            try:
                os.chdir(os.path.dirname(abs_mjcf_path))
                mj_model = mujoco.MjModel.from_xml_path(os.path.basename(abs_mjcf_path))
            finally:
                os.chdir(cwd)
        else:
            mj_model = mujoco.MjModel.from_xml_path(mjcf_path)

        data_dt = config['data_dt']
        sim_step_size = config['sim_step_size']
        mj_model.opt.timestep = data_dt / sim_step_size
        mj_model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        mj_model.opt.iterations = 1
        mj_model.opt.ls_iterations = 0
        mj_model.opt.tolerance = 0
        mj_model.opt.ls_tolerance = 0
        mj_model.opt.noslip_iterations = 0
        mj_model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
        print(f"  Set timestep: {mj_model.opt.timestep:.6f}s (data_dt={data_dt:.4f}s / sim_step_size={sim_step_size})")

    # Load trajectory
    if args.robot == 'omx':
        data = ev.load_csv_data(args.csv, float(config.get('current_lowpass_alpha', 0.0)))
    elif args.robot == 'franka':
        data = ev.load_csv_data(args.csv, config)  # applies trim and lookahead targets
    else:
        data = ev.load_csv_data(args.csv, config.get('current_source', 'load'),
                                float(config.get('current_lowpass_alpha', 0.0)))
    n_samples = data['n_samples']
    n_steps = n_samples - 1
    print(f"Loaded {n_samples} samples from {os.path.basename(args.csv)}")

    # Build per-step CSV features and initial state
    csv_feats = np.zeros((n_steps, feature_dim), dtype=np.float32)
    if args.robot == 'omx':
        for i in range(n_steps):
            csv_feats[i] = ev.build_features(
                data['goal_positions'][i], data['positions'][i], data['aperture'][i] * 1000.0,
                data['currents'][i], data['velocities'][i], data['volts'][i], data['temps'][i],
                data['goal_aperture'][i])
        init_pos = np.concatenate([data['positions'][0, :4], [data['aperture'][0]]]).astype(np.float32)
    elif args.robot == 'franka':
        for i in range(n_steps):
            csv_feats[i] = ev.build_features(
                data['cmd_positions'][i], data['positions'][i], data['gripper_width'][i],
                data['tau_d'][i], data['velocities'][i], data['motor_pos'][i], data['motor_vel'][i])
        init_pos = np.concatenate([data['positions'][0],
                                   [data['gripper_width'][0] * ev.FINGER_TRAVEL]]).astype(np.float32)
    else:
        for i in range(n_steps):
            csv_feats[i] = ev.build_features(
                data['goal_positions'][i], data['positions'][i], data['currents'][i],
                data['velocities'][i], data['volts'][i], data['temps'][i])
        init_pos = data['positions'][0].astype(np.float32)

    # Zero-padded initial history, same as training/eval when starting at the trajectory beginning
    init_history = jnp.zeros((history_length, feature_dim), dtype=jnp.float32)

    if args.force_only:
        forward_fn, step_fn = _build_force_only(model, config, norm_stats, args.robot)

        print(f"Force-only forward pass over {n_steps} steps (teacher-forced, no simulation)...")
        t0 = time.time()
        tau, force_pred, gate, cond = forward_fn(params, init_history, jnp.array(csv_feats))
        jax.block_until_ready(force_pred)
        print(f"  Done in {time.time() - t0:.2f}s (includes JIT compilation)")
        sim_q = None

        # Per-step latency of the single-step function a deployment loop would call
        csv_feats_jax = jnp.array(csv_feats)
        carry = (init_history, params)
        carry, _ = step_fn(carry, csv_feats_jax[0])  # JIT warmup
        jax.block_until_ready(carry[0])
        n_timed = min(500, n_steps)
        t0 = time.time()
        for i in range(n_timed):
            carry, _ = step_fn(carry, csv_feats_jax[i % n_steps])
        jax.block_until_ready(carry[0])
        latency_ms = (time.time() - t0) / n_timed * 1000.0
        print(f"  Per-step forward latency after warmup: {latency_ms:.2f} ms "
              f"({jax.devices()[0].platform}, {n_timed} steps timed)")
    else:
        mjx_model = mjx.put_model(mj_model)
        mjx_data_single = mjx.put_data(mj_model, mujoco.MjData(mj_model))

        if args.robot == 'omx':
            rollout_fn = _build_rollout_omx(model, mjx_model, mjx_data_single, config, norm_stats)
        elif args.robot == 'franka':
            rollout_fn = _build_rollout_franka(model, mjx_model, mjx_data_single, config,
                                               n_joints, ev.FINGER_TRAVEL)
        else:
            rollout_fn = _build_rollout_so101(model, mjx_model, mjx_data_single, config, norm_stats,
                                              n_joints, ev.VEL_COUNTS_PER_RAD)

        print(f"Rolling out {n_steps} steps...")
        t0 = time.time()
        sim_q, tau, force_pred, gate, cond = rollout_fn(params, jnp.array(init_pos), init_history,
                                                        jnp.array(csv_feats))
        jax.block_until_ready(sim_q)
        print(f"  Done in {time.time() - t0:.2f}s (includes JIT compilation)")
        sim_q = np.array(sim_q)

    tau = np.array(tau)
    force_pred = np.array(force_pred)
    gate = np.array(gate)
    cond = np.array(cond)

    # Ground truth: next CSV position at each step, force at the current step
    force_gt = data['forces'][:n_steps].astype(np.float32)
    has_force = data['has_force']

    if args.robot == 'omx':
        gt_arm = data['positions'][1:n_samples, :4]
        gt_grip = data['aperture'][1:n_samples]
        gt_q = np.concatenate([data['positions'][1:n_samples],
                               gt_grip[:, None], gt_grip[:, None]], axis=1)
    elif args.robot == 'franka':
        # 8 columns: 7 arm joints (rad) + finger joint (m)
        gt_q = np.column_stack([
            data['positions'][1:n_samples],
            data['gripper_width'][1:n_samples] * ev.FINGER_TRAVEL,
        ]).astype(np.float32)
    else:
        gt_q = data['positions'][1:n_samples].astype(np.float32)

    if sim_q is None:
        print(f"\nMAE over {n_steps} steps (no joint MAE in force-only mode):")
    else:
        print(f"\nPer-joint MAE over {n_steps} steps (full trajectory):")
        if args.robot == 'omx':
            mae_joints = np.mean(np.abs(np.rad2deg(sim_q[:, :4] - gt_arm)), axis=0)
            mae_grip = np.mean(np.abs((sim_q[:, 4] - gt_grip) * 1000.0))
            joint_str = " ".join([f"J{j+1}={mae_joints[j]:.2f}°" for j in range(4)])
            print(f"  {joint_str} Grip={mae_grip:.2f}mm")
        elif args.robot == 'franka':
            mae_joints = np.mean(np.abs(np.rad2deg(sim_q[:, :n_joints] - gt_q[:, :n_joints])), axis=0)
            mae_grip = np.mean(np.abs((sim_q[:, 7] - gt_q[:, 7]) * 1000.0))
            joint_str = " ".join([f"J{j+1}={mae_joints[j]:.2f}°" for j in range(n_joints)])
            print(f"  {joint_str} Grip={mae_grip:.2f}mm")
        else:
            mae_joints = np.mean(np.abs(np.rad2deg(sim_q - gt_q)), axis=0)
            joint_str = " ".join([f"J{j+1}={mae_joints[j]:.2f}°" for j in range(n_joints)])
            print(f"  {joint_str}")

    if has_force:
        force_error = force_pred - force_gt
        print(f"  Force={np.mean(np.abs(force_error)):.3f}N (all axes)"
              f" ForceZ={np.mean(np.abs(force_error[:, 2])):.3f}N")
    else:
        print("  No force labels in CSV, skipping force MAE")

    # Save predictions
    out_path = args.out
    if out_path is None:
        stem = os.path.splitext(os.path.basename(args.csv))[0]
        out_path = os.path.join('outputs', f'{stem}_pred.npz')
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if out_path.endswith('.csv'):
        cols = {}
        if args.robot == 'omx':
            if sim_q is not None:
                for j in range(4):
                    cols[f'sim_pos{j+1}'] = sim_q[:, j]
                cols['sim_grip'] = sim_q[:, 4]
            for j in range(4):
                cols[f'gt_pos{j+1}'] = gt_arm[:, j]
            cols['gt_grip'] = gt_grip
        elif args.robot == 'franka':
            if sim_q is not None:
                for j in range(n_joints):
                    cols[f'sim_pos{j+1}'] = sim_q[:, j]
                cols['sim_grip'] = sim_q[:, 7]
            for j in range(n_joints):
                cols[f'gt_pos{j+1}'] = gt_q[:, j]
            cols['gt_grip'] = gt_q[:, 7]
        else:
            if sim_q is not None:
                for j in range(n_joints):
                    cols[f'sim_pos{j+1}'] = sim_q[:, j]
            for j in range(n_joints):
                cols[f'gt_pos{j+1}'] = gt_q[:, j]
        for j in range(tau.shape[1]):
            cols[f'tau{j+1}'] = tau[:, j]
        for a_idx, axis in enumerate(['x', 'y', 'z']):
            cols[f'force_pred_{axis}'] = force_pred[:, a_idx]
        if has_force:
            for a_idx, axis in enumerate(['x', 'y', 'z']):
                cols[f'force_gt_{axis}'] = force_gt[:, a_idx]
        cols['gate'] = gate
        pd.DataFrame(cols).to_csv(out_path, index=False)
    else:
        arrays = dict(gt_q=gt_q, tau=tau,
                      force_pred=force_pred, force_gt=force_gt,
                      gate=gate, cond=cond)
        if sim_q is not None:
            arrays['sim_q'] = sim_q
        np.savez(out_path, **arrays)
    print(f"\nPredictions saved to {out_path}")


if __name__ == "__main__":
    main()
