"""Parameter identification of the simplified serial hand from sampled linkage motion."""
from __future__ import annotations

import numpy as np
from numpy.polynomial import polynomial as npoly

from .kinematics import FingerKinematics, monomials, poly2d_features
from .linkage import LinkageHand, MotionSamples

DEG = 180.0 / np.pi


def fit_poly1d(x, y, deg):
    c = npoly.polyfit(x, y, deg)
    r = npoly.polyval(x, c) - y
    return c, float(np.sqrt(np.mean(r ** 2))), float(np.abs(r).max())


def fit_poly2d(x1, x2, y, deg):
    mons = monomials(deg)
    X = poly2d_features(x1, x2, mons)
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = X @ c - y
    return c, mons, float(np.sqrt(np.mean(r ** 2))), float(np.abs(r).max())


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def detect_dead_centre(K: FingerKinematics, S: MotionSamples, i: int, tol_deg=0.05, tol_unexplained_deg=0.5):
    """Per sample and RSS chain: does the identified branch reproduce the commanded motor angle?

    Past a crank dead centre the linkage folds back (same pose for two motor angles): the *other* branch matches.
    Near the dead centre the inverse is ill-conditioned (dtheta/dq -> inf), so a looser tolerance decides whether a
    sample is explained by the closure model at all.
    """
    ok = S.ok[i]
    N = len(S.theta)
    err_fixed = np.full((N, 2), np.nan)
    err_other = np.full((N, 2), np.nan)
    margin = np.full((N, 2), np.nan)
    k0 = int(np.argmin(np.abs(S.theta).sum(axis=1) + 1e9 * (~ok)))
    branch = np.ones(2)
    for c in range(2):
        for b in (1, -1):
            th = K.rss_inverse(S.q[i, k0, 0], S.q[i, k0, 3], branch=[b, b])[c]
            if abs(_wrap(th - S.theta[k0, c])) * DEG < tol_deg:
                branch[c] = b
                break
    for k in np.where(ok)[0]:
        th_fixed = K.rss_inverse(S.q[i, k, 0], S.q[i, k, 3], branch=branch)
        th_other = K.rss_inverse(S.q[i, k, 0], S.q[i, k, 3], branch=-branch)
        err_fixed[k] = np.abs(_wrap(th_fixed - S.theta[k])) * DEG
        err_other[k] = np.abs(_wrap(th_other - S.theta[k])) * DEG
        margin[k] = K.rss_dead_centre_margin(S.q[i, k, 0], S.q[i, k, 3])
    fold = ok[:, None] & (err_fixed > tol_deg) & (err_other < err_fixed)          # (N,2) chain folded back
    unexplained = ok & (np.fmin(err_fixed, err_other) > tol_unexplained_deg).any(axis=1)
    return branch, fold, unexplained, margin


def identify_finger(H: LinkageHand, S: MotionSamples, i: int, deg_pip=4, deg_P=7, deg_motor=6, inv_margin=0.01):
    g = H.geometry[i]
    K = FingerKinematics(g.to_json())
    branch, fold, unexplained, margin = detect_dead_centre(K, S, i)
    beyond = fold.any(axis=1)
    explained = S.ok[i] & ~unexplained
    valid = explained & ~beyond                    # monotonic workspace of the serial model
    q = S.q[i][valid]
    abd, mcp, pip, qP = q.T
    th = S.theta[valid]
    # forward map theta -> q is smooth through the dead centre: fit it on every explained sample (fold-back included)
    qf = S.q[i][explained]
    thf = S.theta[explained]
    # inverse map q -> theta has a sqrt singularity at the dead centre: fit it away from the singular band
    inv_sel = valid & (np.nanmin(margin, axis=1) > inv_margin)
    qi, thi = S.q[i][inv_sel], S.theta[inv_sel]

    c_pip, rms_pip, max_pip = fit_poly1d(mcp, pip, deg_pip)
    c_P, rms_P, max_P = fit_poly1d(mcp, qP, deg_P)
    c_mcpP, rms_mcpP, max_mcpP = fit_poly1d(qP, mcp, deg_P)
    c_abd, mons, rms_abd, max_abd = fit_poly2d(thf[:, 0], thf[:, 1], qf[:, 0], deg_motor)
    c_mcp, _, rms_mcp, max_mcp = fit_poly2d(thf[:, 0], thf[:, 1], qf[:, 1], deg_motor)
    c_t1, mons_inv, rms_t1, max_t1 = fit_poly2d(qi[:, 0], qi[:, 1], thi[:, 0], deg_motor)
    c_t2, _, rms_t2, max_t2 = fit_poly2d(qi[:, 0], qi[:, 1], thi[:, 1], deg_motor)

    diff, summ = thf[:, 0] - thf[:, 1], thf[:, 0] + thf[:, 1]
    cl_mcp, rms_l_mcp, max_l_mcp = fit_poly1d(diff, qf[:, 1], 1)
    cl_abd, rms_l_abd, max_l_abd = fit_poly1d(summ, qf[:, 0], 1)

    fits = {
        "rss_branch_sign": branch.tolist(),
        "coupling_pip_of_mcp": {"coef_ascending": c_pip.tolist(), "domain_rad": [float(mcp.min()), float(mcp.max())],
                                "rms_deg": rms_pip * DEG, "max_deg": max_pip * DEG},
        "coupling_P_of_mcp": {"coef_ascending": c_P.tolist(), "domain_rad": [float(mcp.min()), float(mcp.max())],
                              "rms_deg": rms_P * DEG, "max_deg": max_P * DEG},
        "coupling_mcp_of_P": {"coef_ascending": c_mcpP.tolist(), "domain_rad": [float(qP.min()), float(qP.max())],
                              "rms_deg": rms_mcpP * DEG, "max_deg": max_mcpP * DEG},
        "motor_to_joint_poly": {"degree": deg_motor, "monomials": mons, "coef_abd": c_abd.tolist(), "coef_mcp": c_mcp.tolist(),
                                "rms_deg": [rms_abd * DEG, rms_mcp * DEG], "max_deg": [max_abd * DEG, max_mcp * DEG],
                                "n_samples": int(explained.sum())},
        "joint_to_motor_poly": {"degree": deg_motor, "monomials": mons_inv, "coef_theta1": c_t1.tolist(), "coef_theta2": c_t2.tolist(),
                                "rms_deg": [rms_t1 * DEG, rms_t2 * DEG], "max_deg": [max_t1 * DEG, max_t2 * DEG],
                                "n_samples": int(inv_sel.sum()), "note": "reference only; use the closed-form inverse"},
        "linear_baseline": {"mcp_of_diff": cl_mcp.tolist(), "abd_of_sum": cl_abd.tolist(),
                            "rms_deg": [rms_l_abd * DEG, rms_l_mcp * DEG], "max_deg": [max_l_abd * DEG, max_l_mcp * DEG]},
        "joint_ranges_rad": {"abd": [float(abd.min()), float(abd.max())], "mcp": [float(mcp.min()), float(mcp.max())],
                             "pip": [float(pip.min()), float(pip.max())], "P": [float(qP.min()), float(qP.max())]},
        "motor_workspace_rad": {"theta1": [float(th[:, 0].min()), float(th[:, 0].max())],
                                "theta2": [float(th[:, 1].min()), float(th[:, 1].max())]},
        "dead_centre": {
            "n_samples_folded": int(beyond.sum()), "n_unexplained": int(unexplained.sum()),
            "fold_starts_at_abs_motor_deg": [float(np.abs(S.theta[fold[:, c], c]).min() * DEG) if fold[:, c].any() else None
                                             for c in range(2)],
            "min_margin_valid": [float(np.nanmin(margin[valid, c])) for c in range(2)],
        },
        "n_valid_samples": int(valid.sum()),
    }

    K = FingerKinematics(g.to_json(), fits)
    inv_err = np.array([_wrap(K.joints_to_motors(a, m) - t) for a, m, t in zip(abd, mcp, th)]) * DEG
    fwd = np.array([K.motors_to_joints(t, exact=True) for t in th])
    fwd_err = (fwd - np.stack([abd, mcp, pip], axis=1)) * DEG
    fwd_poly = np.array([K.motors_to_joints(t, exact=False) for t in th])
    fwdp_err = (fwd_poly - np.stack([abd, mcp, pip], axis=1)) * DEG
    fits["validation_deg"] = {
        "joints_to_motors_closed_form": {"rms": np.sqrt((inv_err ** 2).mean(axis=0)).tolist(), "max": np.abs(inv_err).max(axis=0).tolist()},
        "motors_to_joints_newton": {"rms": np.sqrt((fwd_err ** 2).mean(axis=0)).tolist(), "max": np.abs(fwd_err).max(axis=0).tolist()},
        "motors_to_joints_poly": {"rms": np.sqrt((fwdp_err ** 2).mean(axis=0)).tolist(), "max": np.abs(fwdp_err).max(axis=0).tolist()},
    }
    return fits, valid, margin


def format_report(n: int, g, fits: dict) -> str:
    d = g.diagnostics
    v = fits["validation_deg"]
    jr = fits["joint_ranges_rad"]
    rng = lambda k: f"[{jr[k][0] * DEG:6.1f}, {jr[k][1] * DEG:6.1f}]"  # noqa: E731
    lines = [
        f"finger {n}: valid samples {fits['n_valid_samples']}, folded past a dead centre {fits['dead_centre']['n_samples_folded']} "
        f"(fold starts at |theta1|,|theta2| ~ {fits['dead_centre']['fold_starts_at_abs_motor_deg']} deg), "
        f"unexplained {fits['dead_centre']['n_unexplained']}",
        f"   axis signs {g.signs}   joint ranges [deg] abd {rng('abd')} mcp {rng('mcp')} pip {rng('pip')}",
        f"   four-bar links [mm]: ground {d['fourbar']['ground_G_mm']:.2f}, crank {d['fourbar']['crank_P_mm']:.2f}, "
        f"coupler {d['fourbar']['coupler_D_mm']:.2f}, follower {d['fourbar']['follower_L_mm']:.2f}; "
        f"RSS crank radius {d['rss_crank_radius_mm'][0]:.2f} mm, rod {d['rss_rod_length_mm'][0]:.2f} mm",
        f"   pip = poly4(mcp): rms {fits['coupling_pip_of_mcp']['rms_deg']:.4f} deg, max {fits['coupling_pip_of_mcp']['max_deg']:.4f} deg; "
        f"P = poly(mcp): rms {fits['coupling_P_of_mcp']['rms_deg']:.4f} deg",
        f"   motors->(abd,mcp) poly{fits['motor_to_joint_poly']['degree']}: rms {np.array(fits['motor_to_joint_poly']['rms_deg']).round(3)} deg, "
        f"max {np.array(fits['motor_to_joint_poly']['max_deg']).round(3)} deg;  linear baseline rms {np.array(fits['linear_baseline']['rms_deg']).round(2)} deg",
        f"   (abd,mcp)->motors poly{fits['joint_to_motor_poly']['degree']} (reference only, away from dead centre): rms {np.array(fits['joint_to_motor_poly']['rms_deg']).round(3)} deg",
        f"   closed-form inverse  err: rms {np.array(v['joints_to_motors_closed_form']['rms']).round(4)} max {np.array(v['joints_to_motors_closed_form']['max']).round(4)} deg",
        f"   Newton forward       err: rms {np.array(v['motors_to_joints_newton']['rms']).round(4)} max {np.array(v['motors_to_joints_newton']['max']).round(4)} deg",
        f"   poly forward         err: rms {np.array(v['motors_to_joints_poly']['rms']).round(4)} max {np.array(v['motors_to_joints_poly']['max']).round(4)} deg",
    ]
    return "\n".join(lines)
