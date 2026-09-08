"""Motion comparison between the linkage samples and the generated serial model."""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from .kinematics import SerialHandKinematics
from .mathutil import mat_to_quat, quat_to_mat

OBJ = mujoco.mjtObj
DEG = 180.0 / np.pi


def rot_angle_deg(R1, R2):
    c = np.clip((np.trace(R1.T @ R2) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


class SerialModel:
    """Thin wrapper around the generated serial MJCF."""

    def __init__(self, scene_xml: str | Path):
        self.scene_xml = Path(scene_xml)
        self.model = mujoco.MjModel.from_xml_path(str(self.scene_xml))
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.jadr = np.array([[m.jnt_qposadr[mujoco.mj_name2id(m, OBJ.mjOBJ_JOINT, f"f{n}_{j}")] for j in ("abd", "mcp", "pip")]
                              for n in (1, 2, 3, 4)])
        self.act = np.array([[mujoco.mj_name2id(m, OBJ.mjOBJ_ACTUATOR, f"f{n}_{j}") for j in ("abd", "mcp")] for n in (1, 2, 3, 4)])
        self.tip = [mujoco.mj_name2id(m, OBJ.mjOBJ_SITE, f"tip{n}") for n in (1, 2, 3, 4)]
        self.bodies = {(n, k): mujoco.mj_name2id(m, OBJ.mjOBJ_BODY, f"f{n}_{name}")
                       for n in (1, 2, 3, 4) for k, name in (("G", "gimbal"), ("L", "proximal"), ("D", "distal"))}
        palm = m.body_parentid[self.bodies[(1, "G")]]
        mujoco.mj_forward(m, self.data)
        self.palm_pos, self.palm_R = self.data.xpos[palm].copy(), quat_to_mat(self.data.xquat[palm])

    def set_joints(self, q):
        """q (4,3) [abd, mcp, pip] per finger -> forward kinematics (no dynamics)."""
        q = np.asarray(q, float).reshape(4, 3)
        self.data.qpos[self.jadr] = q
        mujoco.mj_forward(self.model, self.data)

    def tip_pose(self, n):
        d = self.data
        return d.site_xpos[self.tip[n - 1]].copy(), d.site_xmat[self.tip[n - 1]].reshape(3, 3).copy()

    def body_pose(self, n, key):
        b = self.bodies[(n, key)]
        return self.data.xpos[b].copy(), quat_to_mat(self.data.xquat[b])


def compare_with_linkage(H, S, valid_by_finger: dict, K: SerialHandKinematics, serial: SerialModel, return_raw=False):
    """Tip pose error of the serial model against the linkage samples, for three ways of obtaining the joints."""
    pW, RW = H._zero["palm"]
    methods = ("exact_joints", "motors_newton", "motors_poly")
    res = {mth: {n: {"pos_mm": [], "ori_deg": []} for n in (1, 2, 3, 4)} for mth in methods}
    body_err = {n: {k: [] for k in "GLD"} for n in (1, 2, 3, 4)}
    fk_err = {n: [] for n in (1, 2, 3, 4)}
    N = len(S.theta)
    for k in range(N):
        theta = S.theta[k]
        q_exact = S.q[:, k, :3]
        q_newton = K.motors_to_joints(np.tile(theta, 4), exact=True)
        q_poly = K.motors_to_joints(np.tile(theta, 4), exact=False)
        for mth, q in zip(methods, (q_exact, q_newton, q_poly)):
            serial.set_joints(q)
            for i, n in enumerate((1, 2, 3, 4)):
                if not valid_by_finger[n][k]:
                    continue
                p_s, R_s = serial.tip_pose(n)
                p_l = S.tip[i, k, :3]
                R_l = quat_to_mat(S.tip[i, k, 3:])
                # express both in the palm frame of their own model (identical palm placement, but be safe)
                p_s = serial.palm_R.T @ (p_s - serial.palm_pos)
                R_s = serial.palm_R.T @ R_s
                p_l = RW.T @ (p_l - pW)
                R_l = RW.T @ R_l
                res[mth][n]["pos_mm"].append(1e3 * np.linalg.norm(p_s - p_l))
                res[mth][n]["ori_deg"].append(rot_angle_deg(R_s, R_l))
                if mth == "exact_joints":
                    for key in "GLD":
                        pb, Rb = serial.body_pose(n, key)
                        pb = serial.palm_R.T @ (pb - serial.palm_pos)
                        pl = RW.T @ (S.pose[key][i, k, :3] - pW)
                        body_err[n][key].append(1e3 * np.linalg.norm(pb - pl))
                    # numpy FK (runtime API) vs MuJoCo FK of the generated model
                    fk = K.fingers[n].fk(*q_exact[i])
                    fk_err[n].append(1e3 * np.linalg.norm(fk["tip"][0] - p_s))
    summary = {}
    for mth in methods:
        summary[mth] = {}
        for n in (1, 2, 3, 4):
            p, o = np.array(res[mth][n]["pos_mm"]), np.array(res[mth][n]["ori_deg"])
            summary[mth][n] = {"pos_rms_mm": float(np.sqrt((p ** 2).mean())), "pos_max_mm": float(p.max()),
                               "ori_rms_deg": float(np.sqrt((o ** 2).mean())), "ori_max_deg": float(o.max()), "n": int(len(p))}
    summary["body_pos_max_mm_exact_joints"] = {n: {k: float(np.max(body_err[n][k])) for k in "GLD"} for n in (1, 2, 3, 4)}
    summary["numpy_fk_vs_mujoco_max_mm"] = {n: float(np.max(fk_err[n])) for n in (1, 2, 3, 4)}
    if return_raw:
        raw = {mth: {n: {k: np.array(v) for k, v in res[mth][n].items()} for n in (1, 2, 3, 4)} for mth in methods}
        return summary, raw
    return summary


def dynamic_coupling_check(serial: SerialModel, K: SerialHandKinematics, seconds=1.0, targets_deg=((0, 30), (20, -40), (-20, 10))) -> dict:
    """Step the serial model with its position servos and verify pip follows the polynomial coupling."""
    m, d = serial.model, serial.data
    out = []
    for abd_deg, mcp_deg in targets_deg:
        mujoco.mj_resetDataKeyframe(m, d, 0)
        target = np.radians([abd_deg, mcp_deg])
        n_steps = int(seconds / m.opt.timestep)
        for s in range(n_steps):
            a = min(1.0, 2.0 * (s + 1) / n_steps)
            d.ctrl[serial.act[:, 0]] = a * target[0]
            d.ctrl[serial.act[:, 1]] = a * target[1]
            mujoco.mj_step(m, d)
        q = d.qpos[serial.jadr]
        pip_expected = np.array([K.fingers[n].pip_of_mcp(q[n - 1, 1]) for n in (1, 2, 3, 4)])
        out.append({"target_deg": [abd_deg, mcp_deg],
                    "abd_err_deg": float(np.abs(q[:, 0] - target[0]).max() * DEG),
                    "mcp_err_deg": float(np.abs(q[:, 1] - target[1]).max() * DEG),
                    "pip_coupling_err_deg": float(np.abs(q[:, 2] - pip_expected).max() * DEG),
                    "max_qvel": float(np.abs(d.qvel).max())})
    mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    return out


def format_validation(summary: dict, dyn: list) -> str:
    lines = ["tip pose error of the serial model vs. the linkage (per finger, over valid samples):"]
    for mth, label in (("exact_joints", "extracted joint angles     "), ("motors_newton", "motors -> Newton closure   "),
                       ("motors_poly", "motors -> polynomial only  ")):
        cells = [f"f{n}: {summary[mth][n]['pos_rms_mm']:.3f}/{summary[mth][n]['pos_max_mm']:.3f} mm, "
                 f"{summary[mth][n]['ori_rms_deg']:.3f}/{summary[mth][n]['ori_max_deg']:.3f} deg" for n in (1, 2, 3, 4)]
        lines.append(f"  {label} (rms/max)  " + " | ".join(cells))
    be = summary["body_pos_max_mm_exact_joints"]
    lines.append("  body origin max error [mm] (G/L/D): " + " | ".join(f"f{n}: {be[n]['G']:.3f}/{be[n]['L']:.3f}/{be[n]['D']:.3f}" for n in (1, 2, 3, 4)))
    lines.append("  numpy FK vs MuJoCo FK max [mm]: " + " ".join(f"{summary['numpy_fk_vs_mujoco_max_mm'][n]:.4f}" for n in (1, 2, 3, 4)))
    lines.append("dynamic check of the serial model (position servos + polynomial coupling equality):")
    for r in dyn:
        lines.append(f"  target abd/mcp {r['target_deg']} deg -> tracking err abd {r['abd_err_deg']:.3f} mcp {r['mcp_err_deg']:.3f} deg, "
                     f"pip coupling err {r['pip_coupling_err_deg']:.3f} deg, max |qvel| {r['max_qvel']:.2e}")
    return "\n".join(lines)
