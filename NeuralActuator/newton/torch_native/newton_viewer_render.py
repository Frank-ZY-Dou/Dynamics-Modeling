"""Render an OMX rollout with the Newton viewer (geometry and renderer both
Newton), using the same side-by-side layout and camera as the MuJoCo renderer.

Two single-arm passes are rendered and hstacked: left = prediction (white arm,
sim_q from the Newton rollout), right = ground truth (green arm, recorded
motion). Each arm sits on a ground plane with a red force arrow at its gripper
(predicted vs ground-truth external force, vertical component for payload tasks).
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("PYGLET_HEADLESS", "1")

import warp as wp
import newton
from newton.viewer import ViewerGL

from backends.base import ROBOT_XML

NDOF = 6
EE = 6
ARROW = (1.0, 0.1, 0.1)
ARROW_SCALE = 0.022
FLOOR_MARGIN = 0.005
WHITE = (1.0, 1.0, 1.0)
GREEN = (0.0, 1.0, 0.0)
CAM = dict(lookat=np.array([0.25, 0.0, 0.15]), distance=1.15, azimuth=90.0, elevation=-6.0, fovy=45.0)


def paper_camera(cam):
    az, el = np.radians(cam["azimuth"]), np.radians(cam["elevation"])
    d = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    eye = cam["lookat"] - cam["distance"] * d
    return eye, float(np.degrees(np.arcsin(d[2]))), float(np.degrees(np.arctan2(d[1], d[0])))


def q6(r):
    return np.array([r[0], r[1], r[2], r[3], r[4], r[4]], np.float32)


def build_model(arm_color):
    xml = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ROBOT_XML))
    b = newton.ModelBuilder()
    b.add_mjcf(xml, parse_meshes=True, parse_visuals=True, enable_self_collisions=False,
               skip_equality_constraints=True, collapse_fixed_joints=False)
    b.add_ground_plane(color=(0.12, 0.12, 0.15))
    model = b.finalize()
    col = model.shape_color.numpy().copy()
    col[0:8] = arm_color
    model.shape_color.assign(col)
    return model


def aim_camera(viewer):
    eye, pitch, yaw = paper_camera(CAM)
    viewer.camera.fov = CAM["fovy"]
    viewer.set_camera(wp.vec3(*eye.tolist()), pitch, yaw)


def force_arrow(ee, f, axis):
    vec = f.astype(np.float64) * ARROW_SCALE
    if axis == "z":
        vec = np.array([0.0, 0.0, vec[2]])
    if np.linalg.norm(vec) < 0.05 * ARROW_SCALE:
        return None
    a = ee.astype(np.float64).copy()
    tip = a + vec
    if tip[2] < FLOOR_MARGIN and vec[2] < 0:
        frac = min(max((a[2] - FLOOR_MARGIN) / (-vec[2]), 0.0), 1.0)
        vec = vec * frac
        if np.linalg.norm(vec) < 0.01:
            return None
    return a, a + vec


def render_pass(viewer, model, poses, forces, args):
    state = model.state()
    jq = state.joint_q.numpy()
    out = []

    def one(pose6, force, t):
        jq[:NDOF] = pose6
        state.joint_q.assign(jq)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        viewer.begin_frame(t)
        viewer.log_state(state)
        if force is not None:
            ee = state.body_q.numpy()[EE, :3]
            ar = force_arrow(ee, force, args.force_axis)
            if ar is not None:
                viewer.log_arrows("force", wp.array(np.array([ar[0]]), dtype=wp.vec3),
                                  wp.array(np.array([ar[1]]), dtype=wp.vec3),
                                  wp.array(np.array([ARROW]), dtype=wp.vec3), width=0.01)
        viewer.end_frame()
        img = viewer.get_frame()
        a = np.array(img.numpy() if hasattr(img, "numpy") else img)
        return a[..., :3] if a.shape[-1] == 4 else a

    one(poses[0], forces[0] if forces is not None else None, 0.0)  # warmup
    dt = args.frame_skip / args.fps
    t = 0.0
    for k in range(0, len(poses), args.frame_skip):
        out.append(one(poses[k], forces[k] if forces is not None else None, t))
        t += dt
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--panel_width", type=int, default=960)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--frame_skip", type=int, default=2)
    ap.add_argument("--force_axis", default="z", choices=["z", "xyz"])
    args = ap.parse_args()

    data = np.load(args.npz)
    sim_q = np.asarray(data["sim_q"])
    gt_q = np.asarray(data["gt_q"])
    gt5 = np.concatenate([gt_q[:, :4], gt_q[:, 5:6]], axis=1)
    fp = np.asarray(data["force_pred"]) if "force_pred" in data.files else None
    fg = np.asarray(data["force_gt"]) if "force_gt" in data.files else None

    # one GL context for both passes (a second headless context renders black);
    # swap the model to recolor between prediction and ground truth
    viewer = ViewerGL(width=args.panel_width, height=args.height, headless=True, vsync=False)
    white_model = build_model(WHITE)
    viewer.set_model(white_model)
    aim_camera(viewer)
    left = render_pass(viewer, white_model, [q6(r) for r in sim_q], fp, args)   # prediction
    green_model = build_model(GREEN)
    viewer.set_model(green_model)
    aim_camera(viewer)
    right = render_pass(viewer, green_model, [q6(r) for r in gt5], fg, args)     # ground truth
    viewer.close()
    n = min(len(left), len(right))
    frames = [np.hstack([left[k], right[k]]) for k in range(n)]

    arr = np.stack(frames)
    if arr.dtype != np.uint8:
        arr = np.clip(arr * (255.0 if arr.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
    import imageio
    imageio.mimwrite(args.out, arr, fps=args.fps, quality=8, macro_block_size=None)
    print(f"wrote {args.out}: {len(frames)} frames {arr.shape[1:]} (Newton viewer)")


if __name__ == "__main__":
    main()
