"""Runtime kinematics of the simplified serial hand, driven by the identified parameter JSON.

Per finger (all in the palm frame, MuJoCo conventions):
    palm ─abd─ G ─mcp─ L ─pip─ D (tip site)         serial chain (what the generated MJCF contains)
    G ─P─ crank                                     the four-bar crank, used only for the motor mapping
    horn_i(theta_i) ─rod_i─ socket_i(crank)         two RSS chains: |socket_i - horn_ball_i| = rod_len_i

motors -> joints : polynomial guess, then Newton on the two RSS closure equations (exact)
joints -> motors : closed form, one trigonometric equation per RSS chain
pip = f(mcp), P = g(mcp), mcp = g^-1(P): 1-D polynomials fitted on the sampled linkage motion
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.polynomial import polynomial as npoly

from .mathutil import quat_to_mat, relative_pose, rot_axis, unit

JOINT_NAMES = ("abd", "mcp", "pip")


def _hinge_body(p_par, R_par, rel_pos, rel_R, jpos, jaxis, q):
    """Pose of a body attached to its parent by one hinge (MuJoCo: rotation about the anchor, axis in body frame)."""
    R_pre = R_par @ rel_R
    p_pre = p_par + R_par @ rel_pos
    anchor = p_pre + R_pre @ jpos
    R = R_pre @ rot_axis(jaxis, q)
    return anchor - R @ jpos, R


def monomials(degree: int):
    return [(i, j) for d in range(degree + 1) for i in range(d, -1, -1) for j in [d - i]]


def poly2d_features(x, y, mons):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return np.stack([x ** i * y ** j for i, j in mons], axis=-1)


class FingerKinematics:
    def __init__(self, geometry: dict, fits: dict | None = None):
        g = {k: (np.asarray(v, float) if isinstance(v, list) else v) for k, v in geometry.items()}
        self.n = int(g["n"])
        RG, RL, RD, RP = (quat_to_mat(g[k + "_quat"]) for k in "GLDP")
        self.G_rel = (g["G_pos"], RG)
        self.L_rel = (lambda p, q: (p, quat_to_mat(q)))(*relative_pose(g["G_pos"], RG, g["L_pos"], RL))
        self.D_rel = (lambda p, q: (p, quat_to_mat(q)))(*relative_pose(g["L_pos"], RL, g["D_pos"], RD))
        self.P_rel = (lambda p, q: (p, quat_to_mat(q)))(*relative_pose(g["G_pos"], RG, g["P_pos"], RP))
        self.abd = (g["abd_jpos"], unit(g["abd_axis"]))
        self.mcp = (g["mcp_jpos"], unit(g["mcp_axis"]))
        self.pip = (g["pip_jpos"], unit(g["pip_axis"]))
        self.Pj = (g["P_jpos"], unit(g["P_axis"]))
        self.c, self.u, self.b0, self.eP, self.rod_len = g["c"], g["u"], g["b0"], g["eP"], g["rod_len"]
        self.tip_pos, self.tip_R = g["tip_pos"], quat_to_mat(g["tip_quat"])
        # RSS chain constants
        self.r_par, self.r_perp, self.w = [], [], []
        for i in range(2):
            r0 = self.b0[i] - self.c[i]
            r_par = (r0 @ self.u[i]) * self.u[i]
            r_perp = r0 - r_par
            self.r_par.append(r_par)
            self.r_perp.append(r_perp)
            self.w.append(np.cross(self.u[i], r_perp))
        self.fits = fits or {}
        self.branch = np.asarray(self.fits.get("rss_branch_sign", [1, 1]), float)
        self._I3 = np.eye(3)
        self._o3 = np.zeros(3)

    # ---------------------------------------------------------------- 1-D couplings
    def _poly(self, key, x):
        f = self.fits.get(key)
        if f is None:
            raise KeyError(f"coupling {key!r} not identified yet")
        return npoly.polyval(np.asarray(x, float), np.asarray(f["coef_ascending"], float))

    def pip_of_mcp(self, mcp):
        return self._poly("coupling_pip_of_mcp", mcp)

    def P_of_mcp(self, mcp):
        return self._poly("coupling_P_of_mcp", mcp)

    def mcp_of_P(self, qP):
        return self._poly("coupling_mcp_of_P", qP)

    # ---------------------------------------------------------------- forward kinematics (palm frame)
    def fk(self, abd, mcp, pip=None):
        if pip is None:
            pip = self.pip_of_mcp(mcp)
        pG, RG = _hinge_body(self._o3, self._I3, *self.G_rel, *self.abd, abd)
        pL, RL = _hinge_body(pG, RG, *self.L_rel, *self.mcp, mcp)
        pD, RD = _hinge_body(pL, RL, *self.D_rel, *self.pip, pip)
        return {"G": (pG, RG), "L": (pL, RL), "D": (pD, RD), "tip": (pD + RD @ self.tip_pos, RD @ self.tip_R)}

    def fk_P(self, abd, qP):
        pG, RG = _hinge_body(self._o3, self._I3, *self.G_rel, *self.abd, abd)
        return _hinge_body(pG, RG, *self.P_rel, *self.Pj, qP)

    def sockets(self, abd, qP):
        pP, RP = self.fk_P(abd, qP)
        return np.stack([pP + RP @ self.eP[i] for i in range(2)])

    def horn_ball(self, i, theta):
        return self.c[i] + self.r_par[i] + np.cos(theta) * self.r_perp[i] + np.sin(theta) * self.w[i]

    # ---------------------------------------------------------------- RSS closure
    def rss_inverse(self, abd, qP, branch=None, theta_ref=None):
        """Motor angles (rad) for a crank configuration: A cos(t) + B sin(t) = C per chain, closed form.

        Each chain has two solutions (mirror images about the dead centre); `theta_ref` selects the one closest
        to a reference (previous command / polynomial estimate), otherwise the identified `branch` sign is used.
        """
        branch = self.branch if branch is None else np.asarray(branch, float)
        e = self.sockets(abd, qP)
        theta = np.zeros(2)
        for i in range(2):
            dvec = e[i] - self.c[i] - self.r_par[i]
            A, B = dvec @ self.r_perp[i], dvec @ self.w[i]
            C = 0.5 * (dvec @ dvec + self.r_perp[i] @ self.r_perp[i] - self.rod_len[i] ** 2)
            R = np.hypot(A, B)
            phi, delta = np.arctan2(B, A), np.arccos(np.clip(C / R, -1.0, 1.0))
            if theta_ref is None:
                theta[i] = phi + branch[i] * delta
            else:
                cands = np.array([phi + delta, phi - delta])
                dist = np.abs((cands - theta_ref[i] + np.pi) % (2 * np.pi) - np.pi)
                theta[i] = cands[np.argmin(dist)]
        return (theta + np.pi) % (2 * np.pi) - np.pi

    def rss_dead_centre_margin(self, abd, qP):
        """acos argument margin per chain (1 - |C/R|); 0 means the crank is exactly at its dead centre."""
        e = self.sockets(abd, qP)
        out = np.zeros(2)
        for i in range(2):
            dvec = e[i] - self.c[i] - self.r_par[i]
            A, B = dvec @ self.r_perp[i], dvec @ self.w[i]
            C = 0.5 * (dvec @ dvec + self.r_perp[i] @ self.r_perp[i] - self.rod_len[i] ** 2)
            out[i] = 1.0 - min(abs(C / np.hypot(A, B)), 1.0)
        return out

    def rss_residual(self, x, theta):
        e = self.sockets(x[0], x[1])
        return np.array([(e[i] - self.horn_ball(i, theta[i])) @ (e[i] - self.horn_ball(i, theta[i])) - self.rod_len[i] ** 2
                         for i in range(2)])

    def rss_forward(self, theta, x0=(0.0, 0.0), iters=30, tol=1e-14):
        """(abd, qP) for given motor angles by Newton iteration on the two closure equations."""
        x = np.asarray(x0, float).copy()
        h = 1e-7
        for _ in range(iters):
            F = self.rss_residual(x, theta)
            if F @ F < tol:
                break
            J = np.zeros((2, 2))
            for j in range(2):
                dx = np.zeros(2)
                dx[j] = h
                J[:, j] = (self.rss_residual(x + dx, theta) - self.rss_residual(x - dx, theta)) / (2 * h)
            try:
                x = x - np.linalg.solve(J, F)
            except np.linalg.LinAlgError:
                break
        return x, float(np.sqrt(F @ F))

    # ---------------------------------------------------------------- public maps
    def motors_to_joints_poly(self, theta):
        f = self.fits["motor_to_joint_poly"]
        X = poly2d_features(theta[..., 0], theta[..., 1], [tuple(m) for m in f["monomials"]])
        abd = X @ np.asarray(f["coef_abd"])
        mcp = X @ np.asarray(f["coef_mcp"])
        return abd, mcp

    def motors_to_joints(self, theta, exact=True, x0=None):
        """theta (2,) [rad] -> (abd, mcp, pip) [rad]."""
        theta = np.asarray(theta, float)
        if x0 is None:
            if "motor_to_joint_poly" in self.fits:
                abd, mcp = self.motors_to_joints_poly(theta)
                x0 = (float(abd), float(self.P_of_mcp(mcp)))
            else:
                x0 = (0.0, 0.0)
        if not exact:
            abd, mcp = self.motors_to_joints_poly(theta)
            return np.array([abd, mcp, self.pip_of_mcp(mcp)])
        x, res = self.rss_forward(theta, x0)
        mcp = self.mcp_of_P(x[1])
        return np.array([x[0], mcp, self.pip_of_mcp(mcp)])

    def joints_to_motors_poly(self, abd, mcp):
        f = self.fits["joint_to_motor_poly"]
        X = poly2d_features(abd, mcp, [tuple(m) for m in f["monomials"]])
        return np.stack([X @ np.asarray(f["coef_theta1"]), X @ np.asarray(f["coef_theta2"])], axis=-1)

    def joints_to_motors(self, abd, mcp, theta_ref=None):
        """(abd, mcp) [rad] -> theta (2,) [rad], closed form.

        Default: the identified branch, i.e. the motor angle on the zero-pose side of the crank dead centre (the
        serial model's monotonic workspace). Pass theta_ref (e.g. the previous command) to pick the nearest branch.
        """
        return self.rss_inverse(abd, self.P_of_mcp(mcp), theta_ref=theta_ref)

    def in_motor_workspace(self, abd, mcp, limit_deg=90.0):
        """True if (abd, mcp) is reachable with both motors inside +-limit_deg. The serial joint ranges are only
        the bounding box of the reachable region, so corners of that box may need motor angles beyond the limit."""
        return bool(np.all(np.abs(self.joints_to_motors(abd, mcp)) <= np.radians(limit_deg) + 1e-9))


class SerialHandKinematics:
    """All four fingers from an identified-parameter JSON."""

    def __init__(self, params: dict):
        self.params = params
        self.side = params["side"]
        self.fingers = {int(n): FingerKinematics(fp["geometry"], fp.get("fits")) for n, fp in params["fingers"].items()}

    @classmethod
    def from_json(cls, path: str | Path):
        with open(path) as fh:
            return cls(json.load(fh))

    def motors_to_joints(self, theta8, exact=True):
        """theta8 (8,) motor angles in the demo order [f1m1,f1m2,...,f4m2] -> (4,3) [abd,mcp,pip] per finger."""
        theta8 = np.asarray(theta8, float).reshape(4, 2)
        return np.stack([self.fingers[n].motors_to_joints(theta8[n - 1], exact=exact) for n in (1, 2, 3, 4)])

    def joints_to_motors(self, q):
        """q (4,2) or (4,3) [abd, mcp(, pip)] -> theta8 (8,)."""
        q = np.asarray(q, float)
        return np.concatenate([self.fingers[n].joints_to_motors(q[n - 1, 0], q[n - 1, 1]) for n in (1, 2, 3, 4)])

    def qpos_from_motors(self, theta8, exact=True):
        return self.qpos_from_joints(self.motors_to_joints(theta8, exact=exact))

    def in_motor_workspace(self, q, limit_deg=90.0):
        """Per-finger reachability of (abd, mcp) targets within the motor limits -> (4,) bool."""
        q = np.asarray(q, float)
        return np.array([self.fingers[n].in_motor_workspace(q[n - 1, 0], q[n - 1, 1], limit_deg) for n in (1, 2, 3, 4)])

    def qpos_from_joints(self, q):
        """(4,2|3) -> serial-model qpos (12,) in the generated MJCF joint order [f1_abd,f1_mcp,f1_pip,...]."""
        q = np.asarray(q, float)
        out = []
        for n in (1, 2, 3, 4):
            abd, mcp = q[n - 1, 0], q[n - 1, 1]
            pip = q[n - 1, 2] if q.shape[1] > 2 else self.fingers[n].pip_of_mcp(mcp)
            out += [abd, mcp, pip]
        return np.array(out)
