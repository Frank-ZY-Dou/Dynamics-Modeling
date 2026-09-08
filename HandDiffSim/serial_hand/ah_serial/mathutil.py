"""Small rotation / geometry helpers (MuJoCo quaternion convention w,x,y,z)."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def quat_to_mat(q_wxyz) -> np.ndarray:
    w, x, y, z = q_wxyz
    return Rotation.from_quat([x, y, z, w]).as_matrix()


def mat_to_quat(R: np.ndarray) -> np.ndarray:
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    q = np.array([w, x, y, z])
    return q if q[0] >= 0 else -q


def rot_axis(axis, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    return Rotation.from_rotvec(axis / np.linalg.norm(axis) * angle).as_matrix()


def unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def hinge_angle(R_rel: np.ndarray, axis_local) -> tuple[float, float]:
    """Angle of a (near) pure rotation R_rel about axis_local, and the off-axis residual (rad).

    R_rel is expressed in the frame where axis_local is defined (the child body frame at zero).
    """
    rv = Rotation.from_matrix(R_rel).as_rotvec()
    a = unit(axis_local)
    q = float(rv @ a)
    resid = float(np.linalg.norm(rv - q * a))
    return q, resid


def line_distance(p1, d1, p2, d2) -> tuple[float, float]:
    """Shortest distance between two lines and the angle (rad) between their directions."""
    d1, d2 = unit(d1), unit(d2)
    n = np.cross(d1, d2)
    ang = float(np.arctan2(np.linalg.norm(n), abs(d1 @ d2)))
    if np.linalg.norm(n) < 1e-9:  # parallel
        w = np.asarray(p2) - np.asarray(p1)
        return float(np.linalg.norm(w - (w @ d1) * d1)), ang
    return float(abs((np.asarray(p2) - np.asarray(p1)) @ n) / np.linalg.norm(n)), ang


def relative_pose(p_parent, R_parent, p_child, R_child):
    """Child pose expressed in the parent frame -> (pos, quat_wxyz)."""
    p_rel = R_parent.T @ (np.asarray(p_child) - np.asarray(p_parent))
    R_rel = R_parent.T @ R_child
    return p_rel, mat_to_quat(R_rel)


def fmt(v, nd=4):
    return np.array2string(np.asarray(v, dtype=float), precision=nd, suppress_small=True, separator=", ")
