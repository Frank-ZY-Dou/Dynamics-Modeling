#!/usr/bin/env python
"""Follow a reference fingertip motion with commands optimized through the differentiable simulator.

The reference is a recorded motion of the original linkage hand (fingertip positions at 50 Hz). Two ways of
turning it into servo commands for the simplified hand are compared:

  kinematic IK   every frame solved on its own with the torch forward kinematics; the commands are the joint
                 angles that place the fingertips on the reference at that instant. Applied to a hand with
                 servo dynamics they lag and cut corners.
  optimized      the same commands, then refined by gradient descent through the simulator: the whole
                 command sequence is optimized so that the *simulated* fingertip trajectory follows the
                 reference, which makes the commands lead the motion and compensate gravity and servo lag.

Both are rolled out open loop through the simplified hand (position-servo mode of HandWarpBackend) from the
recorded initial state, and the fingertip tracking error against the reference is reported.

The optimization uses multiple shooting: the segment is cut into windows that start from the recorded state at
their first step and are simulated as one batch, so an iteration costs one window's worth of steps instead of
the whole segment. The reported error always comes from a single sequential open-loop rollout of the stitched
command sequence.

    python -m diffsim.examples.track_reference --steps 200 --window 25 --iters 150
    -> diffsim/outputs/track/track_summary.json, track_curves.png, track_commands.npz
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from ..backend import HandWarpBackend  # noqa: E402
from ..kinematics_torch import HandKinematicsTorch  # noqa: E402
from ..servo import with_pip  # noqa: E402
from .ik_through_sim import cosine_lr  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
DEG = 180 / np.pi


def kinematic_ik(fk, tips_ref, dev, iters=300, lr=0.05):
    """All frames at once: q8 (N,8) such that FK(q) puts the fingertips on tips_ref (N,4,3)."""
    N = tips_ref.shape[0]
    q8 = torch.zeros(N, 8, device=dev, requires_grad=True)
    opt = torch.optim.Adam([q8], lr=lr)
    for it in range(iters):
        cosine_lr(opt, it, iters, lr)
        q12 = with_pip(q8, fk.pip_of_mcp(q8.reshape(N, 4, 2)[:, :, 1]))
        tips, _ = fk(q12)
        loss = ((tips - tips_ref) ** 2).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        q12 = with_pip(q8, fk.pip_of_mcp(q8.reshape(N, 4, 2)[:, :, 1]))
        tips, _ = fk(q12)
        err = (tips - tips_ref).norm(dim=-1) * 1e3
    return q8.detach(), float(err.mean())


def rollout(be, fk, q0, v0, cmd):
    """Open-loop rollout of the simplified hand under a (N,8) command sequence; returns tips (N,4,3), q12 (N,12)."""
    q, v = q0, v0
    tips, qs = [], []
    for k in range(cmd.shape[0]):
        q, v = be.step(q, v, cmd[k:k + 1])
        t, _ = fk(q)
        tips.append(t[0]); qs.append(q[0])
    return torch.stack(tips), torch.stack(qs)


def rollout_windows(be, fk, q0w, v0w, cmd, L):
    """Batched rollout of W windows of L steps: q0w, v0w (W,12) recorded states at the window starts,
    cmd (W*L, 8) -> tips (W*L, 4, 3) in sequence order."""
    W = q0w.shape[0]
    q, v = q0w, v0w
    tips = []
    for k in range(L):
        q, v = be.step(q, v, cmd.reshape(W, L, 8)[:, k])
        t, _ = fk(q)
        tips.append(t)
    return torch.stack(tips, dim=1).reshape(W * L, 4, 3)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(HERE / "data" / "right" / "val_000.npz"))
    ap.add_argument("--start", type=int, default=0, help="first recording frame of the reference segment")
    ap.add_argument("--steps", type=int, default=200, help="length of the segment in control steps (50 Hz)")
    ap.add_argument("--speed", type=int, default=2, help="playback speed of the recording (integer factor)")
    ap.add_argument("--window", type=int, default=25, help="multiple-shooting window length in steps")
    ap.add_argument("--iters", type=int, default=120, help="windowed optimization iterations")
    ap.add_argument("--refine-iters", type=int, default=20, help="final iterations on the full sequential rollout")
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--smooth", type=float, default=0.01, help="weight of the command-smoothness term (deg^2)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = torch.device(args.device)
    out = HERE / "outputs" / "track"
    out.mkdir(parents=True, exist_ok=True)

    z = np.load(args.data)
    dt = float(z["dt"])
    N, L = args.steps, args.window
    assert N % L == 0, "steps must be a multiple of the window length"
    W = N // L
    idx = args.start + args.speed * np.arange(N)                                 # recording frames used as reference
    assert idx[-1] < len(z["tips"]), "segment runs past the end of the recording"
    tips_ref = torch.from_numpy(z["tips"][idx]).float().to(dev)                 # (N,4,3) reference fingertips
    q_rec = torch.from_numpy(z["q"]).float().to(dev)
    v_rec = torch.from_numpy(np.gradient(z["q"], dt, axis=0)).float().to(dev) * args.speed   # joint velocity at playback speed
    q0, v0 = q_rec[idx[0]:idx[0] + 1], v_rec[idx[0]:idx[0] + 1]                  # recorded initial joint state
    q0w, v0w = q_rec[idx[::L]], v_rec[idx[::L]]                                  # recorded states at the window starts
    fk = HandKinematicsTorch("right", device=dev)
    n_sub = int(round(dt / 0.002))
    be = HandWarpBackend(1, dt, n_sub, device=str(dev), actuation="position")   # sequential rollouts
    bew = HandWarpBackend(W, dt, n_sub, device=str(dev), actuation="position")  # batched windows
    lo = torch.as_tensor(be.mjm.actuator_ctrlrange[:, 0], dtype=torch.float32, device=dev)
    hi = torch.as_tensor(be.mjm.actuator_ctrlrange[:, 1], dtype=torch.float32, device=dev)

    t0 = time.time()
    cmd_kin, fit_mm = kinematic_ik(fk, tips_ref, dev)
    print(f"kinematic IK on {N} frames: fingertip fit {fit_mm:.3f} mm ({time.time() - t0:.1f}s)", flush=True)
    with torch.no_grad():
        tips_kin, q_kin = rollout(be, fk, q0, v0, cmd_kin.clamp(lo, hi))
    err_kin = (tips_kin - tips_ref).norm(dim=-1) * 1e3                            # (N,4)
    print(f"simulated hand under the kinematic-IK commands: tracking error mean {err_kin.mean():.3f} mm, "
          f"max {err_kin.max():.3f} mm", flush=True)

    # stage 1: multiple shooting. Commands stay inside the control range through a sigmoid.
    frac = ((cmd_kin - lo) / (hi - lo)).clamp(0.02, 0.98)
    zz = torch.log(frac / (1 - frac)).clone().requires_grad_(True)
    to_cmd = lambda z_: lo + (hi - lo) * torch.sigmoid(z_)  # noqa: E731

    def loss_of(tips, cmd):
        track = ((tips - tips_ref) ** 2).sum(dim=-1).mean() * 1e6                 # mm^2
        smooth = ((cmd[1:] - cmd[:-1]) ** 2).mean() * DEG ** 2                    # deg^2
        return track + args.smooth * smooth

    opt = torch.optim.Adam([zz], lr=args.lr)
    hist = []
    t0 = time.time()
    for it in range(args.iters):
        cosine_lr(opt, it, args.iters, args.lr)
        cmd = to_cmd(zz)
        tips = rollout_windows(bew, fk, q0w, v0w, cmd, L)
        loss = loss_of(tips, cmd)
        opt.zero_grad(); loss.backward(); opt.step()
        e = float(((tips.detach() - tips_ref).norm(dim=-1) * 1e3).mean())
        hist.append(e)
        if it % 10 == 0 or it == args.iters - 1:
            print(f"  window stage iter {it:3d}: tracking error {e:.3f} mm  [{time.time() - t0:.0f}s]", flush=True)
    # stage 2: a few iterations on the full sequential rollout, so window boundaries are consistent
    opt = torch.optim.Adam([zz], lr=0.3 * args.lr)
    t0 = time.time()
    for it in range(args.refine_iters):
        cosine_lr(opt, it, args.refine_iters, 0.3 * args.lr)
        cmd = to_cmd(zz)
        tips, _ = rollout(be, fk, q0, v0, cmd)
        loss = loss_of(tips, cmd)
        opt.zero_grad(); loss.backward(); opt.step()
        e = float(((tips.detach() - tips_ref).norm(dim=-1) * 1e3).mean())
        hist.append(e)
        if it % 5 == 0 or it == args.refine_iters - 1:
            print(f"  sequential stage iter {it:3d}: tracking error {e:.3f} mm  [{time.time() - t0:.0f}s]", flush=True)
    with torch.no_grad():
        cmd_opt = to_cmd(zz)
        tips_opt, q_opt = rollout(be, fk, q0, v0, cmd_opt)
    err_opt = (tips_opt - tips_ref).norm(dim=-1) * 1e3
    print(f"simulated hand under the optimized commands: tracking error mean {err_opt.mean():.3f} mm, "
          f"max {err_opt.max():.3f} mm", flush=True)

    np.savez(out / "track_commands.npz", dt=dt, start=args.start, steps=N, speed=args.speed, idx=idx, data=str(args.data),
             cmd_kin=cmd_kin.cpu().numpy(), cmd_opt=cmd_opt.cpu().numpy(),
             q_kin=q_kin.cpu().numpy(), q_opt=q_opt.cpu().numpy(), tips_ref=tips_ref.cpu().numpy(),
             tips_kin=tips_kin.cpu().numpy(), tips_opt=tips_opt.cpu().numpy(),
             err_kin=err_kin.cpu().numpy(), err_opt=err_opt.cpu().numpy())
    summary = dict(steps=N, dt=dt, window=L, speed=args.speed, kinematic_ik_fit_mm=fit_mm,
                   tracking_error_mm=dict(kinematic_ik=dict(mean=float(err_kin.mean()), max=float(err_kin.max())),
                                          optimized=dict(mean=float(err_opt.mean()), max=float(err_opt.max()))),
                   iterations=dict(windowed=args.iters, sequential=args.refine_iters), history_mm=hist)
    with open(out / "track_summary.json", "w") as fh:
        json.dump(summary, fh, indent=1)

    t = np.arange(N) * dt
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(t, err_kin.mean(dim=1).cpu().numpy(), color="#eb6834", lw=1.5, label=f"kinematic IK commands (mean {err_kin.mean():.2f} mm)")
    axes[0].plot(t, err_opt.mean(dim=1).cpu().numpy(), color="#2a78d6", lw=1.5, label=f"commands optimized through the simulator (mean {err_opt.mean():.2f} mm)")
    axes[0].set_ylabel("fingertip tracking error [mm]")
    axes[0].legend(frameon=False, fontsize=9)
    axes[0].set_title(f"Following a recorded fingertip motion played {args.speed}x faster, simplified hand", loc="left", fontsize=10.5)
    axes[0].grid(True, color="#e1e0d9")
    j = 1  # finger 1 knuckle
    axes[1].plot(t, z["q"][idx, j] * DEG, color="#0b0b0b", lw=1.5, label=f"reference (recording at {args.speed}x speed)")
    axes[1].plot(t, q_kin[:, j].cpu().numpy() * DEG, color="#eb6834", lw=1.2, label="simulated, kinematic IK commands")
    axes[1].plot(t, q_opt[:, j].cpu().numpy() * DEG, color="#2a78d6", lw=1.2, label="simulated, optimized commands")
    axes[1].plot(t, cmd_opt[:, j].cpu().numpy() * DEG, color="#2a78d6", lw=0.9, ls=(0, (3, 3)), label="optimized command itself")
    axes[1].set_ylabel("finger 1 mcp [deg]")
    axes[1].set_xlabel("time [s]")
    axes[1].legend(frameon=False, fontsize=8, ncol=2)
    axes[1].grid(True, color="#e1e0d9")
    fig.tight_layout()
    fig.savefig(out / "track_curves.png", dpi=140)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
