#!/usr/bin/env python
"""Motion comparison figures and stills: closed-chain linkage vs. the simplified serial model.

Reads params/identified_<side>.json and params/samples_<side>.npz written by build_serial_model.py and writes
figures/*.png (coupling curve, motor map, tip-error CDF, differential-drive baseline) and renders/compare_<side>.png.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from ah_serial.kinematics import FingerKinematics, SerialHandKinematics  # noqa: E402
from ah_serial.linkage import LinkageHand  # noqa: E402
from ah_serial.render import FONT_CJK, Offscreen, make_camera  # noqa: E402
from ah_serial.validate import SerialModel, compare_with_linkage  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent / "AmazingHand"
SCENES = {"right": REPO / "Demo/AHSimulation/AHSimulation/AH_Right/mjcf/scene.xml",
          "left": REPO / "Demo/AHSimulation/AHSimulation/AH_Left/mjcf/scene.xml"}
DEG = 180 / np.pi

# reference palette (dataviz skill): categorical slots, sequential blue ramp, diverging blue<->red, chrome ink
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
SEQ = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]          # ordinal steps 250..650
DIV = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def setup_style():
    try:
        font_manager.fontManager.addfont(FONT_CJK)
        families = ["DejaVu Sans", "Droid Sans Fallback"]
    except Exception:
        families = ["DejaVu Sans"]
    plt.rcParams.update({
        "font.family": families, "font.size": 10.5, "axes.titlesize": 12, "axes.labelsize": 10.5,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": 9.5,
        "lines.linewidth": 1.8, "axes.axisbelow": True,
    })


def load(side):
    with open(HERE / "params" / f"identified_{side}.json") as fh:
        params = json.load(fh)
    npz = np.load(HERE / "params" / f"samples_{side}.npz")
    return params, {k: npz[k] for k in npz.files}


def fig_coupling(side, params, S, out):
    f1 = params["fingers"]["1"]
    K = FingerKinematics(f1["geometry"], f1["fits"])
    v = S["valid"][0]
    mcp, pip, qP = S["q"][0, v, 1] * DEG, S["q"][0, v, 2] * DEG, S["q"][0, v, 3] * DEG
    order = np.argsort(mcp)
    mcp, pip, qP = mcp[order], pip[order], qP[order]
    grid = np.linspace(mcp.min(), mcp.max(), 300)
    pip_fit = K.pip_of_mcp(np.radians(grid)) * DEG
    resid = (K.pip_of_mcp(np.radians(mcp)) * DEG - pip)

    fig, (ax, axr) = plt.subplots(2, 1, figsize=(7.2, 6.4), sharex=True, gridspec_kw={"height_ratios": [3, 1.15], "hspace": 0.08})
    ax.plot(mcp, qP, color=C3, lw=1.6, label="crank P angle (linkage samples)")
    ax.plot(mcp, pip, color=C1, lw=2.2, label="PIP angle (linkage samples)")
    ax.plot(grid, pip_fit, color=C2, lw=1.4, ls=(0, (4, 3)), label="PIP = 4th-order polynomial of MCP (used in the serial MJCF)")
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.axvline(0, color=AXIS, lw=0.8)
    ax.set_ylabel("coupled joint angle [deg]")
    ax.set_title(f"Four-bar coupling: PIP and crank P vs MCP ({side} hand, finger 1)", loc="left")
    ax.legend(loc="upper left")
    ax.annotate("zero pose (CAD keyframe)", xy=(0, 0), xytext=(6, -22), textcoords="offset points", color=INK2, fontsize=9)
    axr.plot(mcp, resid, color=C2, lw=1.4)
    axr.axhline(0, color=AXIS, lw=0.8)
    axr.set_ylabel("fit residual [deg]")
    axr.set_xlabel("MCP angle [deg] (positive = flexion)")
    axr.text(0.99, 0.9, f"rms {f1['fits']['coupling_pip_of_mcp']['rms_deg']:.3f}°, max {f1['fits']['coupling_pip_of_mcp']['max_deg']:.3f}°",
             transform=axr.transAxes, ha="right", va="top", color=INK2, fontsize=9)
    fig.savefig(out / f"coupling_{side}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_motor_map(side, params, S, out):
    theta = S["theta"] * DEG
    g = np.unique(theta[:, 0])
    n = len(g)
    valid = S["valid"][0].reshape(n, n)
    q = S["q"][0] * DEG
    mcp = np.where(S["ok"][0], q[:, 1], np.nan).reshape(n, n)
    abd = np.where(S["ok"][0], q[:, 0], np.nan).reshape(n, n)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))
    for ax, Z, title in ((axes[0], mcp, "MCP flexion [deg]"), (axes[1], abd, "abduction/adduction [deg]")):
        vmax = np.nanmax(np.abs(Z))
        pc = ax.pcolormesh(g, g, Z.T, cmap=DIV, vmin=-vmax, vmax=vmax, shading="nearest", rasterized=True)
        cs = ax.contour(g, g, Z.T, levels=np.arange(-90, 91, 10), colors=INK2, linewidths=0.6, alpha=0.7)
        ax.clabel(cs, fmt="%.0f", fontsize=8, colors=INK2)
        fold = (~valid).astype(float)
        if fold.any():
            ax.contourf(g, g, fold.T, levels=[0.5, 1.5], colors="none", hatches=["////"])
            ax.contour(g, g, fold.T, levels=[0.5], colors=[MUTED], linewidths=0.8)
        ax.set_xlabel("motor 1 angle θ1 [deg]")
        ax.set_ylabel("motor 2 angle θ2 [deg]")
        ax.set_title(title, loc="left")
        ax.set_aspect("equal")
        ax.grid(False)
        cb = fig.colorbar(pc, ax=ax, fraction=0.046, pad=0.03)
        cb.outline.set_edgecolor(AXIS)
        cb.ax.tick_params(color=MUTED, labelcolor=MUTED)
    axes[0].text(0.02, 0.02, "hatched: past the crank dead centre (finger folds back)", transform=axes[0].transAxes,
                 fontsize=8.5, color=INK2, va="bottom")
    fig.suptitle(f"Motor angles to equivalent serial joint angles ({side} hand, finger 1, 5° grid)", x=0.01, ha="left", color=INK, fontsize=12)
    fig.savefig(out / f"motor_map_{side}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_tip_error(side, raw, out):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    labels = {"exact_joints": "joint angles extracted from the linkage", "motors_newton": "motors → RSS closure (Newton) → serial model",
              "motors_poly": "motors → 6th-order polynomial → serial model"}
    for (mth, lab), col in zip(labels.items(), (C1, C2, C3)):
        e = np.sort(np.concatenate([raw[mth][n]["pos_mm"] for n in (1, 2, 3, 4)]))
        e = np.maximum(e, 1e-4)
        y = np.arange(1, len(e) + 1) / len(e)
        ax.plot(e, y, color=col, lw=2.0, label=lab)
        ax.annotate(f"max {e[-1]:.3f} mm", xy=(e[-1], 1.0), xytext=(4, -10 if mth != "motors_poly" else 4),
                    textcoords="offset points", color=col, fontsize=8.5, va="center")
    ax.set_xscale("log")
    ax.set_xlim(1e-4, 2)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("fingertip position error, serial vs linkage [mm] (log scale)")
    ax.set_ylabel("cumulative fraction")
    ax.set_title(f"Motion agreement over all valid samples × 4 fingers ({side} hand)", loc="left")
    ax.legend(loc="upper left")
    fig.savefig(out / f"tip_error_{side}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_linear_baseline(side, params, S, out):
    theta = S["theta"] * DEG
    ok = S["ok"][0] & S["valid"][0]
    q = S["q"][0] * DEG
    diff, summ = (theta[:, 0] - theta[:, 1]) / 2, (theta[:, 0] + theta[:, 1]) / 2
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    sums = [-60, -30, 0, 30, 60]
    for s_val, col in zip(sums, SEQ):
        sel = ok & (np.abs(summ - s_val) < 1e-6)
        o = np.argsort(diff[sel])
        ax.plot(diff[sel][o], q[sel, 1][o], color=col, lw=1.8, label=f"(θ1+θ2)/2 = {s_val:+d}°")
    lin = params["fingers"]["1"]["fits"]["linear_baseline"]["mcp_of_diff"]
    xg = np.linspace(diff[ok].min(), diff[ok].max(), 50)
    ax.plot(xg, (lin[0] + lin[1] * np.radians(2 * xg)) * DEG, color=INK2, lw=1.2, ls=(0, (4, 3)), label="linear differential baseline MCP ≈ a + b·(θ1−θ2)")
    ax.set_xlabel("differential command (θ1 − θ2)/2 [deg]")
    ax.set_ylabel("MCP flexion [deg]")
    ax.set_title(f"Flexion and abduction are coupled: MCP vs differential command depends on the common-mode command ({side} hand, finger 1)", loc="left", fontsize=10.5)
    ax.legend(loc="upper left", ncol=2)
    rms = params["fingers"]["1"]["fits"]["linear_baseline"]["rms_deg"]
    ax.text(0.99, 0.03, f"linear baseline residual rms: abd {rms[0]:.2f}°, mcp {rms[1]:.2f}°", transform=ax.transAxes, ha="right", color=INK2, fontsize=9)
    fig.savefig(out / f"linear_baseline_{side}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def stills(side, H, K, serial, out, size=420):
    """Linkage (top row) vs serial (bottom row) at the same motor commands."""
    from PIL import Image, ImageDraw
    from ah_serial.render import draw_text
    poses = [("zero", (0, 0)), ("flexion", (85, -85)), ("extension", (-85, 85)),
             ("abduction", (35, 35)), ("adduction", (-35, -35)), ("mixed", (60, -20))]
    m, d = H.model, H.data
    m.opt.gravity[:] = 0
    m.dof_frictionloss[:] = 0
    act = np.array([[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"finger{n}_motor{i}") for i in (1, 2)] for n in (1, 2, 3, 4)])
    cam = make_camera(lookat=(0.03, 0.0, 0.10), distance=0.30, azimuth=160, elevation=-20)
    r_link, r_ser = Offscreen(m, (size, size), cam), Offscreen(serial.model, (size, size), cam)
    top, bottom, errs = [], [], []
    for label, (t1, t2) in poses:
        H.reset_zero()
        th = np.radians([t1, t2])
        for s in range(400):
            a = min(1.0, (s + 1) / 250)
            d.ctrl[act[:, 0]], d.ctrl[act[:, 1]] = a * th[0], a * th[1]
            mujoco.mj_step(m, d)
        mujoco.mj_forward(m, d)
        q = K.motors_to_joints(np.tile(th, 4), exact=True)
        serial.set_joints(q)
        errs.append(max(1e3 * np.linalg.norm(d.site_xpos[f.tip_s] - serial.data.site_xpos[serial.tip[i]]) for i, f in enumerate(H.fingers)))
        top.append(r_link.render(d))
        bottom.append(r_ser.render(serial.data))
    r_link.close()
    r_ser.close()
    H.reset_zero()
    W, Hh, header, rowlab = size, size, 46, 30
    canvas = Image.new("RGB", (len(poses) * W + 150, header + 2 * Hh + 2 * rowlab), (18, 20, 26))
    draw = ImageDraw.Draw(canvas)
    draw_text(draw, (12, 10), f"Same motor commands: top = original closed-chain linkage, bottom = simplified serial model ({side} hand)", 22)
    for j, ((label, (t1, t2)), e) in enumerate(zip(poses, errs)):
        x0 = 150 + j * W
        canvas.paste(Image.fromarray(top[j]), (x0, header + rowlab))
        canvas.paste(Image.fromarray(bottom[j]), (x0, header + 2 * rowlab + Hh))
        draw_text(draw, (x0 + 8, header + 4), f"{label}  θ=({t1:+d}°, {t2:+d}°)", 17, fill=(220, 220, 220))
        draw_text(draw, (x0 + 8, header + rowlab + Hh + 4), f"fingertip error {e:.3f} mm", 17, fill=(220, 220, 220))
    draw_text(draw, (10, header + rowlab + Hh // 2 - 10), "linkage", 17, fill=(220, 220, 220))
    draw_text(draw, (10, header + 2 * rowlab + Hh + Hh // 2 - 10), "serial", 17, fill=(220, 220, 220))
    canvas.save(out / f"compare_{side}.png")
    return errs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--side", choices=["right", "left", "both"], default="both")
    args = ap.parse_args()
    setup_style()
    fig_dir, ren_dir = HERE / "figures", HERE / "renders"
    fig_dir.mkdir(exist_ok=True)
    ren_dir.mkdir(exist_ok=True)
    for side in (["right", "left"] if args.side == "both" else [args.side]):
        params, S = load(side)
        H = LinkageHand(SCENES[side])
        K = SerialHandKinematics(params)
        serial = SerialModel(HERE / "models" / f"AH_{side.capitalize()}" / "scene.xml")
        # rebuild a MotionSamples-like object for compare_with_linkage
        from ah_serial.linkage import MotionSamples
        MS = MotionSamples(S["theta"], S["ok"], S["closure_mm"], np.zeros_like(S["closure_mm"]),
                           {k: S[f"pose_{k}"] for k in "GPLD"}, S["tip"], S["q"], S["resid"])
        valid = {n: S["valid"][n - 1] for n in (1, 2, 3, 4)}
        _, raw = compare_with_linkage(H, MS, valid, K, serial, return_raw=True)
        fig_coupling(side, params, S, fig_dir)
        fig_motor_map(side, params, S, fig_dir)
        fig_tip_error(side, raw, fig_dir)
        fig_linear_baseline(side, params, S, fig_dir)
        errs = stills(side, H, K, serial, ren_dir)
        print(f"{side}: figures written to {fig_dir.relative_to(HERE.parent)}, stills to renders/compare_{side}.png "
              f"(tip errors at the 6 poses: {np.round(errs, 3)} mm)")


if __name__ == "__main__":
    main()
