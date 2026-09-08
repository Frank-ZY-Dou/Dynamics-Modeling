#!/usr/bin/env python
"""Inverse kinematics by gradient descent through the differentiable simulator.

Given a target position for each fingertip, find the inputs that bring the simulated hand there. Three ways
to pose the problem, all solved with Adam on the gradients that HandWarpBackend and the torch forward
kinematics provide:

  torque     unknowns are the eight joint torques at every step of a 0.6 s horizon (torque -> pose: the
             torques are applied through the backend, the hand starts at rest, and the loss is the fingertip
             error over the last steps plus small effort and terminal-velocity penalties)
  command    unknowns are eight constant servo targets; the model's own position servos (kp = 50) act inside
             MuJoCo Warp at every substep and the loss is the fingertip error once the hand has settled
  kinematic  unknowns are the eight driven joint angles, no simulator, torch forward kinematics only (reference)

The targets are generated from a known joint configuration, so the recovered joint angles can be compared with
the ground truth as well as the fingertip error.

Saturation kills gradients: once a servo command leaves the actuator's control range or a torque hits its
bound, the simulated pose stops responding to that parameter and its finite-difference gradient is exactly
zero, so the parameter never comes back. Commands are therefore parametrized as lo + (hi - lo) sigmoid(z) and
torques as tau_max tanh(z), and the torque mode adds a mild penalty for approaching the joint limits. Local
minima remain possible, so the sim-based modes run several random starts in parallel (one world each; the
simulator is batched) and keep, per finger, the start with the smallest fingertip error. The assembled
solution is re-simulated once so the reported error is its own.

    python -m diffsim.examples.ik_through_sim --mode torque command kinematic --starts 12
    -> diffsim/outputs/ik/ik_summary.json, ik_curves.png, ik_pose_<mode>.png
"""
import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from ..backend import HandWarpBackend  # noqa: E402
from ..kinematics_torch import HandKinematicsTorch  # noqa: E402
from ..paths import SERIAL_DIR  # noqa: E402
from ..servo import with_pip  # noqa: E402

sys.path.insert(0, str(SERIAL_DIR))
from ah_serial.render import Offscreen, make_camera  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
DEG = 180 / np.pi

# ground-truth joint configuration the targets come from: (abd, mcp) per finger in degrees
Q_TRUE_DEG = np.array([[-6.0, 30.0], [2.0, 42.0], [8.0, 35.0], [12.0, 18.0]])


def targets_from(fk, q8_deg, dev):
    q8 = torch.tensor(np.radians(q8_deg).reshape(1, 8), dtype=torch.float32, device=dev)
    q12 = with_pip(q8, fk.pip_of_mcp(q8.reshape(1, 4, 2)[:, :, 1]))
    tips, _ = fk(q12)
    return tips.detach(), q12.detach()


def cosine_lr(opt, it, iters, lr0, floor=0.05):
    """Adam's step stays ~lr until convergence, so decay it to settle below a millimetre."""
    for g in opt.param_groups:
        g["lr"] = lr0 * (floor + (1 - floor) * 0.5 * (1 + np.cos(np.pi * it / iters)))


def tip_error_mm(tips, targets):
    return ((tips - targets).norm(dim=-1) * 1e3).squeeze(0)


def solve_kinematic(fk, targets, dev, iters=400, lr=0.05):
    q8 = torch.zeros(1, 8, device=dev, requires_grad=True)
    opt = torch.optim.Adam([q8], lr=lr)
    hist = []
    for it in range(iters):
        cosine_lr(opt, it, iters, lr)
        q12 = with_pip(q8, fk.pip_of_mcp(q8.reshape(1, 4, 2)[:, :, 1]))
        tips, _ = fk(q12)
        loss = ((tips - targets) ** 2).sum()
        opt.zero_grad(); loss.backward(); opt.step()
        hist.append(float(tip_error_mm(tips.detach(), targets).mean()))
    with torch.no_grad():
        q12 = with_pip(q8, fk.pip_of_mcp(q8.reshape(1, 4, 2)[:, :, 1]))
        tips, _ = fk(q12)
    return q12.detach(), tips.detach(), hist


def _assemble(x, err):
    """Per finger, pick the start (world) with the smallest fingertip error. x: (S, ..., 8) inputs with the last
    axis ordered (f1_abd, f1_mcp, ..., f4_mcp); err: (S, 4). Returns the assembled inputs with a leading 1."""
    best = err.argmin(dim=0)                                        # (4,)
    out = torch.zeros_like(x[:1])
    for n in range(4):
        out[..., 2 * n:2 * n + 2] = x[best[n], ..., 2 * n:2 * n + 2][None]
    return out, best


def solve_command(be, fk, targets, dev, K=30, iters=300, lr=0.1, seed=0):
    """Constant servo targets held for K control steps, S random starts in parallel; loss on the settled tips.
    The command lives inside the actuator control range through a sigmoid, so it can never saturate."""
    S = be.B
    lo = torch.as_tensor(be.mjm.actuator_ctrlrange[:, 0], dtype=torch.float32, device=dev)
    hi = torch.as_tensor(be.mjm.actuator_ctrlrange[:, 1], dtype=torch.float32, device=dev)
    g = torch.Generator(device=dev).manual_seed(seed)
    z = (torch.randn(S, 8, device=dev, generator=g) * 0.7).requires_grad_(True)
    to_cmd = lambda z_: lo + (hi - lo) * torch.sigmoid(z_)  # noqa: E731
    opt = torch.optim.Adam([z], lr=lr)
    q0 = torch.zeros(S, 12, device=dev)
    v0 = torch.zeros(S, 12, device=dev)
    tg = targets.expand(S, 4, 3)
    hist = []

    def run(c):
        q, v = q0, v0
        for k in range(K):
            q, v = be.step(q, v, c)
        return q, v

    for it in range(iters):
        cosine_lr(opt, it, iters, lr)
        q, v = run(to_cmd(z))
        tips, _ = fk(q)
        loss = ((tips - tg) ** 2).sum() + 1e-4 * (v ** 2).sum()
        opt.zero_grad(); loss.backward(); opt.step()
        hist.append(float((tips.detach() - tg).norm(dim=-1).mean(dim=1).min() * 1e3))
    with torch.no_grad():
        cmd = to_cmd(z)
        q, v = run(cmd)
        tips, _ = fk(q)
        err = (tips - tg).norm(dim=-1) * 1e3                              # (S,4)
        best_cmd, best = _assemble(cmd, err)
        q, v = run(best_cmd.expand(S, 8))                                 # re-simulate the assembled solution
        tips, _ = fk(q)
    converged = (err < 1.0).float().mean(dim=0).cpu().numpy()             # fraction of starts within 1 mm, per finger
    return q[:1], tips[:1], hist, best_cmd, converged


def solve_torque(be, fk, targets, dev, K=30, iters=300, lr=0.02, tau_max=0.3, n_last=5, seed=0, trace=None):
    """Torque sequence over K control steps, S random starts in parallel; loss on the last n_last steps.
    Each start begins from a small constant torque (a different one per start); per-step noise would send
    the fingers on erratic paths the optimizer then has to undo."""
    S = be.B
    g = torch.Generator(device=dev).manual_seed(seed)
    z = (torch.randn(1, S, 8, device=dev, generator=g) * 0.05).expand(K, S, 8).clone().requires_grad_(True)
    to_tau = lambda z_: tau_max * torch.tanh(z_)  # noqa: E731
    lo = torch.as_tensor(be.mjm.jnt_range[:, 0], dtype=torch.float32, device=dev)
    hi = torch.as_tensor(be.mjm.jnt_range[:, 1], dtype=torch.float32, device=dev)
    margin = np.radians(3.0)
    opt = torch.optim.Adam([z], lr=lr)
    q0 = torch.zeros(S, 12, device=dev)
    v0 = torch.zeros(S, 12, device=dev)
    tg = targets.expand(S, 4, 3)
    hist = []

    def range_penalty(q):
        """Grows quadratically once a joint is within `margin` of its limit; zero elsewhere."""
        return (torch.relu(q - (hi - margin)) ** 2 + torch.relu((lo + margin) - q) ** 2).sum()

    def run(t, with_loss=False):
        q, v = q0, v0
        loss = 0.0
        for k in range(K):
            q, v = be.step(q, v, t[k])
            if with_loss:
                loss = loss + 10.0 * range_penalty(q) / K
                if k >= K - n_last:
                    tips, _ = fk(q)
                    loss = loss + ((tips - tg) ** 2).sum() / n_last
        return q, v, loss

    for it in range(iters):
        cosine_lr(opt, it, iters, lr)
        tau = to_tau(z)
        q, v, loss = run(tau, with_loss=True)
        tips, _ = fk(q)
        loss = loss + 1e-3 * (v ** 2).sum() + 1e-2 * (tau ** 2).mean() * S
        opt.zero_grad(); loss.backward(); opt.step()
        e_it = (tips.detach() - tg).norm(dim=-1).mean(dim=1)               # (S,) mean error per start
        hist.append(float(e_it.min() * 1e3))
        if trace is not None:
            trace.append((q.detach()[e_it.argmin()].cpu().numpy(), float(e_it.min() * 1e3)))
    with torch.no_grad():
        tau = to_tau(z)
        q, v, _ = run(tau)
        tips, _ = fk(q)
        err = (tips - tg).norm(dim=-1) * 1e3
        best_tau, best = _assemble(tau.permute(1, 0, 2), err)             # starts first: (S,K,8) -> (1,K,8)
        best_tau = best_tau.permute(1, 0, 2)                                # back to (K,1,8)
        q, v, _ = run(best_tau.expand(K, S, 8))
        tips, _ = fk(q)
    converged = (err < 1.0).float().mean(dim=0).cpu().numpy()
    return q[:1], tips[:1], hist, best_tau, converged


def motion_under(be, tau_seq, K):
    """Joint trajectory (K+1, 12) of world 0 from rest under a (K,1,8) torque sequence."""
    S = be.B
    q = torch.zeros(S, 12, device=tau_seq.device)
    v = torch.zeros(S, 12, device=tau_seq.device)
    out = [q[0].cpu().numpy()]
    with torch.no_grad():
        for k in range(K):
            q, v = be.step(q, v, tau_seq[k].expand(S, 8))
            out.append(q[0].cpu().numpy())
    return np.stack(out)


def render_pose(q12, targets, path, title):
    """The simplified hand at q12 with the four target markers, offscreen."""
    mjm = mujoco.MjModel.from_xml_path(str(SERIAL_DIR / "models" / "AH_Right" / "scene.xml"))
    d = mujoco.MjData(mjm)
    d.qpos[:] = q12.cpu().numpy().ravel()
    for n in range(4):
        b = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, f"finger{n + 1}_target")
        d.mocap_pos[mjm.body_mocapid[b]] = targets[0, n].cpu().numpy()
    mujoco.mj_forward(mjm, d)
    r = Offscreen(mjm, (640, 640), make_camera(lookat=(0.03, 0.0, 0.10), distance=0.30, azimuth=160, elevation=-20))
    img = r.render(d)
    r.close()
    from PIL import Image, ImageDraw
    from ah_serial.render import draw_text
    im = Image.fromarray(img)
    draw_text(ImageDraw.Draw(im), (12, 10), title, 20)
    im.save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", nargs="+", default=["torque", "command", "kinematic"], choices=["torque", "command", "kinematic"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--n-sub", type=int, default=10)
    ap.add_argument("--starts", type=int, default=12, help="parallel random starts for the sim-based modes")
    args = ap.parse_args()
    dev = torch.device(args.device)
    out = HERE / "outputs" / "ik"
    out.mkdir(parents=True, exist_ok=True)
    fk = HandKinematicsTorch("right", device=dev)
    targets, q_true = targets_from(fk, Q_TRUE_DEG, dev)
    print("fingertip targets [m] from the ground-truth joints:")
    for n in range(4):
        print(f"  finger {n + 1}: {np.round(targets[0, n].cpu().numpy(), 4)}   (abd, mcp) = {Q_TRUE_DEG[n]} deg")

    summary, curves = {}, {}
    for mode in args.mode:
        t0 = time.time()
        if mode == "kinematic":
            q12, tips, hist = solve_kinematic(fk, targets, dev)
            extra = {}
        elif mode == "command":
            be = HandWarpBackend(args.starts, args.dt, args.n_sub, device=str(dev), actuation="position")
            q12, tips, hist, cmd, conv = solve_command(be, fk, targets, dev)
            extra = {"servo_targets_deg": (cmd.cpu().numpy().ravel() * DEG).round(2).tolist(),
                     "starts": args.starts, "fraction_of_starts_within_1mm_per_finger": conv.round(2).tolist()}
        else:
            be = HandWarpBackend(args.starts, args.dt, args.n_sub, device=str(dev), actuation="torque")
            q12, tips, hist, tau, conv = solve_torque(be, fk, targets, dev)
            extra = {"torque_rms_Nm": float(tau.pow(2).mean().sqrt()), "torque_final_Nm": tau[-1].cpu().numpy().ravel().round(4).tolist(),
                     "starts": args.starts, "fraction_of_starts_within_1mm_per_finger": conv.round(2).tolist()}
        err = tip_error_mm(tips, targets).cpu().numpy()
        q8 = q12.reshape(4, 3)[:, :2].cpu().numpy() * DEG
        joint_err = np.abs(q8 - Q_TRUE_DEG)
        sec = time.time() - t0
        summary[mode] = dict(tip_error_mm=err.round(3).tolist(), tip_error_mean_mm=float(err.mean()),
                             joints_deg=q8.round(2).tolist(), joint_error_max_deg=float(joint_err.max()),
                             iterations=len(hist), seconds=round(sec, 1), **extra)
        curves[mode] = hist
        print(f"{mode:9s}: mean fingertip error {err.mean():.3f} mm (per finger {np.round(err, 3)}), "
              f"max joint error {joint_err.max():.2f} deg, {len(hist)} iterations in {sec:.1f}s"
              + (f", starts within 1 mm per finger {extra['fraction_of_starts_within_1mm_per_finger']}" if "starts" in extra else ""))
        render_pose(q12, targets, out / f"ik_pose_{mode}.png", f"IK via simulator ({mode}): tip error {err.mean():.2f} mm")

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for (mode, hist), col in zip(curves.items(), ("#2a78d6", "#eb6834", "#1baf7a")):
        ax.plot(np.arange(1, len(hist) + 1), hist, color=col, lw=1.8, label=mode)
    ax.set_yscale("log")
    ax.set_xlabel("Adam iteration")
    ax.set_ylabel("mean fingertip error of the best start [mm]")
    ax.set_title("Inverse kinematics by gradient descent through the simulator", loc="left", fontsize=10.5)
    ax.grid(True, color="#e1e0d9")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "ik_curves.png", dpi=140)
    with open(out / "ik_summary.json", "w") as fh:
        json.dump(dict(targets_m=targets.cpu().numpy()[0].round(5).tolist(), q_true_deg=Q_TRUE_DEG.tolist(), modes=summary), fh, indent=1)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
