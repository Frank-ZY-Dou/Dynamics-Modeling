"""Render an OMX rollout npz with MuJoCo's native offscreen renderer (the same
engine family the mjwarp backend simulates), framed like the paper's MJX
renderer: two single-arm passes hstacked, left = prediction (white arm,
mjwarp-advanced sim_q), right = ground truth (green arm, recorded motion), each
with a red force arrow at the gripper (predicted vs ground-truth external
force). Arrow is tail-anchored at the grasp point, length = force * scale with
no floor clamp, so the drawn length reflects the force.
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

import mujoco
import numpy as np

XML = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "newton", "robot", "omx_newton.xml"))

WHITE = np.array([1.0, 1.0, 1.0, 1.0], np.float32)
GREEN = np.array([0.2, 0.9, 0.2, 1.0], np.float32)
ARROW_RGBA = np.array([1.0, 0.1, 0.1, 1.0], np.float32)
# camera and arrow conventions identical to the public render_rollout.py OMX
# preset (the released videos), so renders are consistent across engines
CAM = dict(lookat=[0.25, 0.0, 0.15], distance=1.10, azimuth=90.0, elevation=-5.2)


def q6(row5):
    return np.array([row5[0], row5[1], row5[2], row5[3], row5[4], row5[4]], np.float64)


def _arrow_offset():
    az = np.deg2rad(CAM["azimuth"])
    el = np.deg2rad(CAM["elevation"])
    toward_cam = np.array([-np.cos(az) * np.cos(el), -np.sin(az) * np.cos(el), -np.sin(el)])
    return toward_cam * 0.02


def add_force_arrow(scene, origin, force, scale, axis="z"):
    vec = np.asarray(force, np.float64) * scale
    if axis == "z":
        vec = np.array([0.0, 0.0, vec[2]])
    if np.linalg.norm(vec) < 0.05 * scale:
        return
    if scene.ngeom >= scene.maxgeom:
        return
    a = np.asarray(origin, np.float64) + _arrow_offset()
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3),
                        np.zeros(9), ARROW_RGBA)
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, 0.005,
                         a, a + vec)
    scene.ngeom += 1


def render_pass(mjm, poses, forces, args, color):
    body_rgba = mjm.geom_rgba.copy()
    # recolor the whole robot including the world-attached base plate; only the
    # floor keeps its material (matches the released renders)
    floor_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    robot_geoms = [g for g in range(mjm.ngeom) if g != floor_id]
    for g in robot_geoms:
        mjm.geom_rgba[g] = color
    mjd = mujoco.MjData(mjm)
    ee = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, "end_effector_target")
    ren = mujoco.Renderer(mjm, height=args.height, width=args.panel_width)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = CAM["lookat"]
    cam.distance = CAM["distance"]
    cam.azimuth = CAM["azimuth"]
    cam.elevation = CAM["elevation"]
    frames = []
    for k in range(0, len(poses), args.frame_skip):
        mjd.qpos[:] = q6(poses[k])
        mujoco.mj_forward(mjm, mjd)
        ren.update_scene(mjd, camera=cam)
        if forces is not None:
            add_force_arrow(ren.scene, mjd.xpos[ee].copy(), forces[k],
                            args.arrow_scale, args.force_axis)
        frames.append(ren.render().copy())
    ren.close()
    mjm.geom_rgba[:] = body_rgba
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--panel_width", type=int, default=960)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--frame_skip", type=int, default=2)
    ap.add_argument("--arrow_scale", type=float, default=0.03)
    ap.add_argument("--force_axis", default="z", choices=["z", "xyz"])
    args = ap.parse_args()

    data = np.load(args.npz)
    sim_q = np.asarray(data["sim_q"])                       # (T,5)
    gt_q = np.asarray(data["gt_q"])                         # (T,7)
    gt5 = np.concatenate([gt_q[:, :4], gt_q[:, 5:6]], axis=1)
    fp = np.asarray(data["force_pred"]) if "force_pred" in data.files else None
    fg = np.asarray(data["force_gt"]) if "force_gt" in data.files else None

    # paper-look visual environment (floor / light / skybox from the public
    # robot/scene.xml), merged into the dynamics model via a sibling temp file
    # so relative mesh paths keep resolving
    VISUAL = """
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
      markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>
  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane" contype="0" conaffinity="0"/>
  </worldbody>
</mujoco>"""
    xml_text = open(XML).read().replace("</mujoco>", VISUAL)
    tmp_xml = os.path.join(os.path.dirname(XML), "_render_tmp.xml")
    with open(tmp_xml, "w") as f:
        f.write(xml_text)
    try:
        mjm = mujoco.MjModel.from_xml_path(tmp_xml)
    finally:
        os.remove(tmp_xml)
    mjm.vis.global_.offwidth = max(args.panel_width, mjm.vis.global_.offwidth)
    mjm.vis.global_.offheight = max(args.height, mjm.vis.global_.offheight)
    left = render_pass(mjm, sim_q, fp, args, WHITE)         # prediction
    right = render_pass(mjm, gt5, fg, args, GREEN)          # ground truth
    n = min(len(left), len(right))
    arr = np.stack([np.hstack([left[k], right[k]]) for k in range(n)])

    import imageio
    if args.out.endswith(".gif"):
        imageio.mimwrite(args.out, arr, fps=args.fps)
    else:
        imageio.mimwrite(args.out, arr, fps=args.fps, quality=8, macro_block_size=None)
    print(f"wrote {args.out}: {n} frames {arr.shape[1:]}")


if __name__ == "__main__":
    main()
