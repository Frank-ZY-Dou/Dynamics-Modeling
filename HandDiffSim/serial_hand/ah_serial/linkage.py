"""Probe the official AmazingHand closed-chain MJCF and sample its motion.

Per finger n the onshape-to-robot export contains (names from the official model):

    palm ─motor1─ horn1 ─ball─ rod1 ─ball─ P ─hinge "passive2 (n)"─ G          G: gimbal body (universal joint)
    palm ─motor2─ horn2 ─ball─ rod2 ─connect "closing_ballN"─ P                 P: crank ("link" part, ball sockets)
    P ─hinge "passive4 (n)"─ D ─hinge "passive_5 (n)"─ L ─connect "closing_3"─ G   D: distal phalanx (tip site)
    G ─connect "closing_1 (n)"─ palm                                             L: proximal phalanx

Serial simplification used by this package (per finger):

    palm ─abd (closing_1 axis)─ G ─mcp (closing_3 axis)─ L ─pip (passive_5 axis)─ D

with pip = f(mcp) (four-bar coupling) and the two RSS chains (horn-rod-socket) mapping motors <-> (abd, P).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from .mathutil import hinge_angle, line_distance, mat_to_quat, quat_to_mat, relative_pose, unit

OBJ = mujoco.mjtObj
FINGERS = (1, 2, 3, 4)


@dataclass
class FingerStructure:
    n: int
    motor_j: tuple
    horn_b: tuple
    rod1_b: int
    rod2_b: int
    G: int
    P: int
    L: int
    D: int
    j_P: int
    j_PD: int
    j_pip: int
    ball1_j: int
    ball2_j: int
    ball3_j: int
    s_abd_G: tuple
    s_abd_W: tuple
    s_mcp_L: tuple
    s_mcp_G: tuple
    s_rod2: tuple
    tip_s: int
    eq_ids: list = field(default_factory=list)


@dataclass
class FingerGeometry:
    """Zero-pose geometry. Poses are in the PALM frame; hinge definitions in the child body frame."""
    n: int
    G_pos: np.ndarray
    G_quat: np.ndarray
    L_pos: np.ndarray
    L_quat: np.ndarray
    D_pos: np.ndarray
    D_quat: np.ndarray
    P_pos: np.ndarray
    P_quat: np.ndarray
    abd_jpos: np.ndarray
    abd_axis: np.ndarray
    mcp_jpos: np.ndarray
    mcp_axis: np.ndarray
    pip_jpos: np.ndarray
    pip_axis: np.ndarray
    P_jpos: np.ndarray
    P_axis: np.ndarray
    c: np.ndarray        # (2,3) point on motor axis i (palm frame)
    u: np.ndarray        # (2,3) motor axis direction (palm frame, positive motor angle = right-hand rule)
    b0: np.ndarray       # (2,3) horn ball centre at zero (palm frame)
    eP: np.ndarray       # (2,3) socket ball centre on the crank P (P frame)
    rod_len: np.ndarray  # (2,)
    tip_pos: np.ndarray
    tip_quat: np.ndarray
    diagnostics: dict = field(default_factory=dict)
    signs: dict = field(default_factory=lambda: {"abd": 1, "mcp": 1, "pip": 1, "P": 1})

    def to_json(self) -> dict:
        out = {}
        for k, v in self.__dict__.items():
            out[k] = v.tolist() if isinstance(v, np.ndarray) else v
        return out


@dataclass
class MotionSamples:
    theta: np.ndarray            # (N,2) commanded motor angles [rad], identical for the 4 fingers
    ok: np.ndarray               # (4,N) converged and closed
    closure_mm: np.ndarray       # (4,N)
    motor_err: np.ndarray        # (4,N) max |actual - commanded| [rad]
    pose: dict                   # name -> (4,N,7) world pos+quat for 'G','P','L','D'
    tip: np.ndarray              # (4,N,7) world pos+quat of tip site
    q: np.ndarray | None = None       # (4,N,4) [abd, mcp, pip, P] (rad)
    resid: np.ndarray | None = None   # (4,N,4) off-axis residual (rad)


class LinkageHand:
    """The official closed-chain model, with structure probing, geometry extraction and motion sampling."""

    def __init__(self, scene_xml: str | Path):
        self.scene_xml = Path(scene_xml).resolve()
        self.robot_xml = self.scene_xml.parent / "robot.xml"
        self.model = mujoco.MjModel.from_xml_path(str(self.scene_xml))
        self.data = mujoco.MjData(self.model)
        self.side = "left" if "left" in self.scene_xml.as_posix().lower() else "right"
        self.key_zero = mujoco.mj_name2id(self.model, OBJ.mjOBJ_KEY, "zero")
        self.fingers = [self._probe_finger(n) for n in FINGERS]
        palms = {self.model.body_parentid[f.horn_b[0]] for f in self.fingers}
        assert len(palms) == 1, "all fingers must hang from the same palm body"
        self.palm = palms.pop()
        self.palm_name = mujoco.mj_id2name(self.model, OBJ.mjOBJ_BODY, self.palm)
        self.reset_zero()
        self._zero = self._snapshot_world()
        self.geometry = [self._extract_geometry(f) for f in self.fingers]

    # ------------------------------------------------------------------ structure
    def _probe_finger(self, n: int) -> FingerStructure:
        m = self.model

        def jid(name):
            i = mujoco.mj_name2id(m, OBJ.mjOBJ_JOINT, name)
            assert i >= 0, f"joint {name!r} not found"
            return i

        def sid(name):
            i = mujoco.mj_name2id(m, OBJ.mjOBJ_SITE, name)
            assert i >= 0, f"site {name!r} not found"
            return i

        motor = (jid(f"finger{n}_motor1"), jid(f"finger{n}_motor2"))
        j_P, j_PD, j_pip = jid(f"passive2 ({n})"), jid(f"passive4 ({n})"), jid(f"passive_5 ({n})")
        G, D, L = int(m.jnt_bodyid[j_P]), int(m.jnt_bodyid[j_PD]), int(m.jnt_bodyid[j_pip])
        P = int(m.body_parentid[G])
        assert m.body_parentid[D] == P and m.body_parentid[L] == D, "unexpected four-bar topology"
        rod1 = int(m.body_parentid[P])
        horn1 = int(m.body_parentid[rod1])
        assert m.jnt_bodyid[motor[0]] == horn1, "motor1 must drive the horn of rod 1"
        horn2 = int(m.jnt_bodyid[motor[1]])
        s_rod2 = (sid(f"closing_ball{n}_1"), sid(f"closing_ball{n}_2"))
        rod2 = int(m.site_bodyid[s_rod2[0]])
        assert m.body_parentid[rod2] == horn2 and m.site_bodyid[s_rod2[1]] == P, "unexpected rod-2 closure"

        def ball_of(b):
            js = [j for j in range(m.njnt) if m.jnt_bodyid[j] == b and m.jnt_type[j] == mujoco.mjtJoint.mjJNT_BALL]
            assert len(js) == 1, f"body {b} should carry exactly one ball joint"
            return js[0]

        s_abd_G = (sid(f"closing_1 ({n})_1"), sid(f"closing_1 ({n})_1_z"))
        s_abd_W = (sid(f"closing_1 ({n})_2"), sid(f"closing_1 ({n})_2_z"))
        s_mcp_L = (sid(f"closing_3 ({n})_1"), sid(f"closing_3 ({n})_1_z"))
        s_mcp_G = (sid(f"closing_3 ({n})_2"), sid(f"closing_3 ({n})_2_z"))
        assert m.site_bodyid[s_abd_G[0]] == G and m.site_bodyid[s_mcp_L[0]] == L and m.site_bodyid[s_mcp_G[0]] == G
        eq_ids = [e for e in range(m.neq)
                  if m.eq_objtype[e] == OBJ.mjOBJ_SITE and m.eq_obj1id[e] in (*s_abd_G, *s_mcp_L, s_rod2[0])]
        assert len(eq_ids) == 5, f"finger {n}: expected 5 connect constraints, found {len(eq_ids)}"
        return FingerStructure(n, motor, (horn1, horn2), rod1, rod2, G, P, L, D, j_P, j_PD, j_pip,
                               ball_of(rod1), ball_of(P), ball_of(rod2), s_abd_G, s_abd_W, s_mcp_L, s_mcp_G,
                               s_rod2, sid(f"tip{n}"), eq_ids)

    # ------------------------------------------------------------------ state helpers
    def reset_zero(self):
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.key_zero)
        mujoco.mj_forward(self.model, self.data)

    def body_pose(self, b):
        return self.data.xpos[b].copy(), quat_to_mat(self.data.xquat[b])

    def _snapshot_world(self) -> dict:
        d = self.data
        snap = {"palm": self.body_pose(self.palm)}
        for f in self.fingers:
            for key, b in (("G", f.G), ("P", f.P), ("L", f.L), ("D", f.D)):
                snap[(f.n, key)] = self.body_pose(b)
        return snap

    def closure_error_mm(self, f: FingerStructure) -> float:
        m, d = self.model, self.data
        return 1e3 * max(np.linalg.norm(d.site_xpos[m.eq_obj1id[e]] - d.site_xpos[m.eq_obj2id[e]]) for e in f.eq_ids)

    # ------------------------------------------------------------------ geometry
    def _extract_geometry(self, f: FingerStructure) -> FingerGeometry:
        m, d = self.model, self.data
        pW, RW = self._zero["palm"]
        to_palm = lambda p: RW.T @ (np.asarray(p) - pW)  # noqa: E731
        (pG, RG), (pP, RP), (pL, RL), (pD, RD) = (self._zero[(f.n, k)] for k in "GPLD")

        # serial hinges, in child-body frames
        abd_jpos = m.site_pos[f.s_abd_G[0]].copy()
        abd_axis = unit(m.site_pos[f.s_abd_G[1]] - m.site_pos[f.s_abd_G[0]])
        mcp_jpos = m.site_pos[f.s_mcp_L[0]].copy()
        mcp_axis = unit(m.site_pos[f.s_mcp_L[1]] - m.site_pos[f.s_mcp_L[0]])
        pip_anchor_w = pL + RL @ m.jnt_pos[f.j_pip]            # passive_5 lives on L in the linkage tree
        pip_axis_w = RL @ m.jnt_axis[f.j_pip]
        pip_jpos = RD.T @ (pip_anchor_w - pD)                   # ... but D is the child in the serial tree
        pip_axis = unit(RD.T @ pip_axis_w)
        P_anchor_w = pG + RG @ m.jnt_pos[f.j_P]                 # passive2 lives on G; P is the child in the RSS model
        P_axis_w = RG @ m.jnt_axis[f.j_P]
        P_jpos = RP.T @ (P_anchor_w - pP)
        P_axis = unit(RP.T @ P_axis_w)

        # RSS chains
        c = np.stack([to_palm(d.xanchor[j]) for j in f.motor_j])
        u = np.stack([unit(RW.T @ d.xaxis[j]) for j in f.motor_j])
        b0 = np.stack([to_palm(d.xanchor[f.ball1_j]), to_palm(d.xanchor[f.ball3_j])])
        e_w = np.stack([d.xanchor[f.ball2_j], d.site_xpos[f.s_rod2[1]]])
        eP = np.stack([RP.T @ (e - pP) for e in e_w])
        rod_len = np.array([np.linalg.norm(e_w[i] - (pW + RW @ b0[i])) for i in range(2)])

        # diagnostics: closure slack of the two hinge closures, universal-joint axes, four-bar link lengths
        def site_line(s):
            p0, p1 = d.site_xpos[s[0]], d.site_xpos[s[1]]
            return p0, unit(p1 - p0)

        abd_G_w, abd_W_w = site_line(f.s_abd_G), site_line(f.s_abd_W)
        mcp_L_w, mcp_G_w = site_line(f.s_mcp_L), site_line(f.s_mcp_G)
        gap_abd, ang_abd = line_distance(*abd_G_w, *abd_W_w)
        gap_mcp, ang_mcp = line_distance(*mcp_L_w, *mcp_G_w)
        uj_dist, uj_ang = line_distance(*abd_W_w, *mcp_G_w)
        ax_P = (d.xanchor[f.j_P], d.xaxis[f.j_P])
        ax_PD = (d.xanchor[f.j_PD], d.xaxis[f.j_PD])
        ax_pip = (d.xanchor[f.j_pip], d.xaxis[f.j_pip])
        ax_mcp = mcp_G_w
        fourbar = {
            "ground_G_mm": 1e3 * line_distance(*ax_P, *ax_mcp)[0],
            "crank_P_mm": 1e3 * line_distance(*ax_P, *ax_PD)[0],
            "coupler_D_mm": 1e3 * line_distance(*ax_PD, *ax_pip)[0],
            "follower_L_mm": 1e3 * line_distance(*ax_pip, *ax_mcp)[0],
            "max_axis_nonparallel_deg": float(np.degrees(max(
                line_distance(*ax_P, *ax_PD)[1], line_distance(*ax_PD, *ax_pip)[1],
                line_distance(*ax_pip, *ax_mcp)[1], line_distance(*ax_P, *ax_mcp)[1]))),
        }
        crank_r = [float(np.linalg.norm((b0[i] - c[i]) - ((b0[i] - c[i]) @ u[i]) * u[i])) for i in range(2)]
        diagnostics = {
            "abd_closure_gap_mm": 1e3 * gap_abd, "abd_closure_angle_deg": float(np.degrees(ang_abd)),
            "mcp_closure_gap_mm": 1e3 * gap_mcp, "mcp_closure_angle_deg": float(np.degrees(ang_mcp)),
            "ujoint_axes_distance_mm": 1e3 * uj_dist, "ujoint_axes_angle_deg": float(np.degrees(uj_ang)),
            "fourbar": fourbar,
            "rss_crank_radius_mm": [1e3 * r for r in crank_r],
            "rss_rod_length_mm": (1e3 * rod_len).tolist(),
            "motor_axes_angle_deg": float(np.degrees(np.arccos(np.clip(abs(u[0] @ u[1]), 0, 1)))),
        }
        Gp, Gq = relative_pose(pW, RW, pG, RG)
        Lp, Lq = relative_pose(pW, RW, pL, RL)
        Dp, Dq = relative_pose(pW, RW, pD, RD)
        Pp, Pq = relative_pose(pW, RW, pP, RP)
        return FingerGeometry(f.n, Gp, Gq, Lp, Lq, Dp, Dq, Pp, Pq, abd_jpos, abd_axis, mcp_jpos, mcp_axis,
                              pip_jpos, pip_axis, P_jpos, P_axis, c, u, b0, eP, rod_len,
                              m.site_pos[f.tip_s].copy(), m.site_quat[f.tip_s].copy(), diagnostics)

    # ------------------------------------------------------------------ sampling
    def sample_motor_grid(self, theta: np.ndarray, ramp_steps=150, hold_steps=250, extra_steps=1500,
                          tol=2e-4, vel_tol=2e-3, closure_tol_mm=1.0, verbose=True) -> MotionSamples:
        """Quasi-static motion of the linkage for each commanded motor pair (same command on all 4 fingers).

        Gravity and joint friction are switched off (kinematic identification); position servos ramp from the
        zero keyframe to the target to stay in the assembly mode of the CAD zero pose.
        """
        m, d = self.model, self.data
        theta = np.asarray(theta, dtype=float).reshape(-1, 2)
        N = len(theta)
        saved = (m.opt.gravity.copy(), m.dof_frictionloss.copy())
        m.opt.gravity[:] = 0.0
        m.dof_frictionloss[:] = 0.0
        act = np.array([[mujoco.mj_name2id(m, OBJ.mjOBJ_ACTUATOR, f"finger{n}_motor{i}") for i in (1, 2)] for n in FINGERS])
        qadr = np.array([[m.jnt_qposadr[j] for j in f.motor_j] for f in self.fingers])
        ok = np.zeros((4, N), bool)
        closure = np.zeros((4, N))
        merr = np.zeros((4, N))
        pose = {k: np.zeros((4, N, 7)) for k in "GPLD"}
        tip = np.zeros((4, N, 7))
        try:
            for k, th in enumerate(theta):
                self.reset_zero()
                for s in range(ramp_steps):
                    a = (s + 1) / ramp_steps
                    d.ctrl[act[:, 0]] = a * th[0]
                    d.ctrl[act[:, 1]] = a * th[1]
                    mujoco.mj_step(m, d)
                d.ctrl[act[:, 0]], d.ctrl[act[:, 1]] = th[0], th[1]
                for s in range(hold_steps):
                    mujoco.mj_step(m, d)
                steps = 0
                while True:
                    err = np.abs(d.qpos[qadr] - th[None, :]).max()
                    if (err < tol and np.abs(d.qvel).max() < vel_tol) or steps >= extra_steps:
                        break
                    mujoco.mj_step(m, d)
                    steps += 1
                mujoco.mj_forward(m, d)
                for i, f in enumerate(self.fingers):
                    merr[i, k] = np.abs(d.qpos[qadr[i]] - th).max()
                    closure[i, k] = self.closure_error_mm(f)
                    ok[i, k] = merr[i, k] < 5 * tol and closure[i, k] < closure_tol_mm and np.abs(d.qvel).max() < vel_tol
                    for key, b in (("G", f.G), ("P", f.P), ("L", f.L), ("D", f.D)):
                        pose[key][i, k, :3] = d.xpos[b]
                        pose[key][i, k, 3:] = d.xquat[b]
                    tip[i, k, :3] = d.site_xpos[f.tip_s]
                    tip[i, k, 3:] = mat_to_quat(d.site_xmat[f.tip_s].reshape(3, 3))
                if verbose and (k % max(1, N // 10) == 0 or k == N - 1):
                    print(f"  sampled {k + 1}/{N}  theta=({np.degrees(th[0]):6.1f},{np.degrees(th[1]):6.1f}) deg  "
                          f"ok={ok[:, k].astype(int)}  closure={closure[:, k].max():.3f} mm", flush=True)
        finally:
            m.opt.gravity[:] = saved[0]
            m.dof_frictionloss[:] = saved[1]
            self.reset_zero()
        samples = MotionSamples(theta, ok, closure, merr, pose, tip)
        self.extract_joint_angles(samples)
        return samples

    # ------------------------------------------------------------------ joint-angle extraction
    def extract_joint_angles(self, s: MotionSamples):
        """Equivalent serial joint angles [abd, mcp, pip, P] from sampled body orientations (MuJoCo convention:
        R_child = R_parent * R_rel0 * Rot(axis_child_local, q))."""
        N = len(s.theta)
        q = np.zeros((4, N, 4))
        resid = np.zeros((4, N, 4))
        for i, (f, g) in enumerate(zip(self.fingers, self.geometry)):
            _, RG0 = self._zero[(f.n, "G")]
            _, RP0 = self._zero[(f.n, "P")]
            _, RL0 = self._zero[(f.n, "L")]
            _, RD0 = self._zero[(f.n, "D")]
            for k in range(N):
                RG = quat_to_mat(s.pose["G"][i, k, 3:])
                RP = quat_to_mat(s.pose["P"][i, k, 3:])
                RL = quat_to_mat(s.pose["L"][i, k, 3:])
                RD = quat_to_mat(s.pose["D"][i, k, 3:])
                q[i, k, 0], resid[i, k, 0] = hinge_angle(RG0.T @ RG, g.abd_axis)
                q[i, k, 1], resid[i, k, 1] = hinge_angle(RL0.T @ RG0 @ RG.T @ RL, g.mcp_axis)
                q[i, k, 2], resid[i, k, 2] = hinge_angle(RD0.T @ RL0 @ RL.T @ RD, g.pip_axis)
                q[i, k, 3], resid[i, k, 3] = hinge_angle(RP0.T @ RG0 @ RG.T @ RP, g.P_axis)
        s.q, s.resid = q, resid

    def apply_sign_conventions(self, s: MotionSamples) -> dict:
        """Flip hinge axes so that: mcp>0 = flexion (tip moves towards the palm side), pip and P increase with mcp,
        abd>0 = tip moves towards the thumb (fingers 1-3) / towards the index finger (thumb). Re-extracts q."""
        d = self.data  # at the zero pose
        tips0 = np.array([d.site_xpos[f.tip_s] for f in self.fingers])
        bases0 = np.array([d.site_xpos[f.s_abd_W[0]] for f in self.fingers])
        finger_axis = unit((tips0[:3] - bases0[:3]).mean(axis=0))            # longitudinal direction of fingers 1-3
        spread = unit(bases0[2] - bases0[0])                                  # index -> ring
        palm_normal = unit(np.cross(finger_axis, spread))
        if (tips0[3] - tips0[:3].mean(axis=0)) @ palm_normal < 0:             # thumb sits on the palm side
            palm_normal = -palm_normal
        thumb_axis = unit(tips0[3] - bases0[3])
        to_fingers = tips0[:3].mean(axis=0) - tips0[3]
        thumb_flex_dir = unit(to_fingers - (to_fingers @ thumb_axis) * thumb_axis)
        report = {}
        for i, (f, g) in enumerate(zip(self.fingers, self.geometry)):
            okk = s.ok[i]
            k = np.argmax(np.abs(s.q[i, :, 1]) * okk)                         # largest |mcp| excursion
            disp = s.tip[i, k, :3] - tips0[i]
            flex_dir = palm_normal if i < 3 else thumb_flex_dir
            if np.sign(s.q[i, k, 1]) * np.sign(disp @ flex_dir) < 0:
                g.mcp_axis = -g.mcp_axis
                g.signs["mcp"] = -1
            for name, col in (("pip", 2), ("P", 3)):                          # coupled joints follow mcp
                slope = np.polyfit(s.q[i, okk, 1], s.q[i, okk, col], 1)[0] * g.signs["mcp"]
                if slope < 0:
                    setattr(g, f"{name}_axis", -getattr(g, f"{name}_axis"))
                    g.signs[name] = -1
            ref = tips0[3] if i < 3 else tips0[0]                             # abduction: towards thumb / index
            k = np.argmax(np.abs(s.q[i, :, 0]) * okk)
            closer = np.linalg.norm(s.tip[i, k, :3] - ref) < np.linalg.norm(tips0[i] - ref)
            if (s.q[i, k, 0] > 0) != closer:
                g.abd_axis = -g.abd_axis
                g.signs["abd"] = -1
            report[f.n] = dict(g.signs)
        self.extract_joint_angles(s)
        self.palm_normal = palm_normal
        return report

    # ------------------------------------------------------------------ reporting
    def describe(self) -> str:
        m = self.model
        name = lambda t, i: mujoco.mj_id2name(m, t, i)  # noqa: E731
        lines = [f"linkage model: {self.scene_xml}  side={self.side}  nbody={m.nbody} njnt={m.njnt} neq={m.neq} nu={m.nu}",
                 f"palm body: {self.palm_name}"]
        for f, g in zip(self.fingers, self.geometry):
            dg = g.diagnostics
            lines += [
                f"finger {f.n}: G={name(OBJ.mjOBJ_BODY, f.G)!r} P={name(OBJ.mjOBJ_BODY, f.P)!r}",
                f"          L={name(OBJ.mjOBJ_BODY, f.L)!r} D={name(OBJ.mjOBJ_BODY, f.D)!r}",
                f"          closure slack at zero: abd {dg['abd_closure_gap_mm']:.3f} mm, mcp {dg['mcp_closure_gap_mm']:.3f} mm; "
                f"U-joint axes: distance {dg['ujoint_axes_distance_mm']:.3f} mm, angle {dg['ujoint_axes_angle_deg']:.2f} deg",
                f"          four-bar [mm] ground(G)={dg['fourbar']['ground_G_mm']:.2f} crank(P)={dg['fourbar']['crank_P_mm']:.2f} "
                f"coupler(D)={dg['fourbar']['coupler_D_mm']:.2f} follower(L)={dg['fourbar']['follower_L_mm']:.2f} "
                f"(axes non-parallel <= {dg['fourbar']['max_axis_nonparallel_deg']:.3f} deg)",
                f"          RSS chains: crank radius {dg['rss_crank_radius_mm'][0]:.2f}/{dg['rss_crank_radius_mm'][1]:.2f} mm, "
                f"rod length {dg['rss_rod_length_mm'][0]:.2f}/{dg['rss_rod_length_mm'][1]:.2f} mm, motor axes angle {dg['motor_axes_angle_deg']:.2f} deg",
            ]
        return "\n".join(lines)
