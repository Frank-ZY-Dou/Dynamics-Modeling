"""OBB-OBB collision oracle using the Separating Axis Theorem (SAT).

Provides SignedDistanceOracle and PenetrationDepthOracle interfaces for
oriented boxes, enabling the baseline solvers to work with the same cube-mesh
geometry used by S4R.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from common import PairPenetration, PairSignedDistance


@dataclass
class BoxObject:
    """An oriented bounding box (OBB)."""
    name: str
    center: np.ndarray       # (3,) world-space centroid
    half_extents: np.ndarray  # (3,) half-widths along local axes
    rotation: np.ndarray      # (3,3) rotation matrix (columns = local axes)
    inv_mass: float = 1.0


def _sat_test(
    c_a: np.ndarray, h_a: np.ndarray, R_a: np.ndarray,
    c_b: np.ndarray, h_b: np.ndarray, R_b: np.ndarray,
) -> tuple[bool, float, np.ndarray]:
    """Separating Axis Theorem for two OBBs.

    Returns
    -------
    is_overlapping : bool
    depth_or_gap : float
        If overlapping: penetration depth (positive).
        If separated: separation gap (positive).
    direction : np.ndarray (3,)
        Unit vector pointing from A toward B along the axis of minimum overlap
        (or maximum separation).
    """
    d = c_b - c_a  # displacement from A center to B center

    # Precompute dot products between local axes
    # R_a columns: a0, a1, a2;  R_b columns: b0, b1, b2
    axes_a = [R_a[:, 0], R_a[:, 1], R_a[:, 2]]
    axes_b = [R_b[:, 0], R_b[:, 1], R_b[:, 2]]

    min_overlap = float("inf")
    best_axis = np.array([1.0, 0.0, 0.0])
    best_sign = 1.0

    def _test_axis(L_raw: np.ndarray) -> bool:
        """Test a single separating axis. Returns False if separated."""
        nonlocal min_overlap, best_axis, best_sign

        length = np.linalg.norm(L_raw)
        if length < 1e-10:
            return True  # degenerate axis, skip
        L = L_raw / length

        proj_d = np.dot(L, d)
        r_a = sum(h_a[k] * abs(np.dot(L, axes_a[k])) for k in range(3))
        r_b = sum(h_b[k] * abs(np.dot(L, axes_b[k])) for k in range(3))
        overlap = r_a + r_b - abs(proj_d)

        if overlap < min_overlap:
            min_overlap = overlap
            best_axis = L.copy()
            best_sign = 1.0 if proj_d >= 0 else -1.0

        return overlap >= 0  # True = still potentially overlapping

    # Test 15 axes
    all_overlap = True

    # 3 face normals of A
    for i in range(3):
        if not _test_axis(axes_a[i]):
            all_overlap = False

    # 3 face normals of B
    for i in range(3):
        if not _test_axis(axes_b[i]):
            all_overlap = False

    # 9 edge cross products
    for i in range(3):
        for j in range(3):
            cross = np.cross(axes_a[i], axes_b[j])
            if not _test_axis(cross):
                all_overlap = False

    # Direction: point from A toward B
    direction = best_sign * best_axis

    if all_overlap:
        # Penetrating: min_overlap > 0 is the penetration depth
        return True, min_overlap, direction
    else:
        # Separated: -min_overlap is the gap (min_overlap < 0)
        return False, -min_overlap, direction


class BoxOracle:
    """Collision oracle for oriented boxes using SAT.

    Implements both SignedDistanceOracle and PenetrationDepthOracle protocols.
    """

    def __init__(self, boxes: List[BoxObject]):
        self.boxes = boxes
        self.n = len(boxes)
        # Cache per-box data
        self._half_extents = [b.half_extents for b in boxes]
        self._rotations = [b.rotation for b in boxes]

    def signed_distances(self, centers: np.ndarray) -> List[PairSignedDistance]:
        """Return signed distances for all pairs.

        signed_distance < 0 means penetration.
        signed_distance > 0 means separation.
        """
        pairs: List[PairSignedDistance] = []
        for i in range(self.n):
            for j in range(i + 1, self.n):
                is_overlap, value, direction = _sat_test(
                    centers[i], self._half_extents[i], self._rotations[i],
                    centers[j], self._half_extents[j], self._rotations[j],
                )
                if is_overlap:
                    sd = -value  # penetration → negative signed distance
                else:
                    sd = value   # separation → positive signed distance
                pairs.append(PairSignedDistance(
                    i=i, j=j,
                    signed_distance=float(sd),
                    direction_ij=direction,
                ))
        return pairs

    def penetrations(self, centers: np.ndarray) -> List[PairPenetration]:
        """Return penetration info for overlapping pairs only."""
        out: List[PairPenetration] = []
        for i in range(self.n):
            for j in range(i + 1, self.n):
                is_overlap, depth, direction = _sat_test(
                    centers[i], self._half_extents[i], self._rotations[i],
                    centers[j], self._half_extents[j], self._rotations[j],
                )
                if is_overlap and depth > 0:
                    out.append(PairPenetration(
                        i=i, j=j,
                        depth=float(depth),
                        direction_ij=direction,
                    ))
        return out


def euler_to_rotation_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    """Convert Euler angles to rotation matrix using LCS convention: Rx * Ry * Rz."""
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    Rx = np.array([
        [1,  0,   0],
        [0,  cx, -sx],
        [0,  sx,  cx],
    ])
    Ry = np.array([
        [ cy, 0, sy],
        [  0, 1,  0],
        [-sy, 0, cy],
    ])
    Rz = np.array([
        [cz, -sz, 0],
        [sz,  cz, 0],
        [ 0,   0, 1],
    ])
    return Rx @ Ry @ Rz


def generate_scene(
    n_boxes: int,
    box_scale: float,
    spawn_range_xz: float,
    spawn_range_y: float,
    seed: int,
) -> List[BoxObject]:
    """Generate a scene of randomly placed oriented cubes.

    Matches the scene generation in S4R's test_resolve_penetration.py exactly:
    same seed, same random call order, same rotation convention.
    """
    rng = np.random.RandomState(seed)
    half = box_scale * 0.5  # half-extent of each cube
    local_centroid = np.array([0.5, 0.5, 0.5]) * box_scale

    boxes: List[BoxObject] = []
    for i in range(n_boxes):
        # Position (translation p in LCS)
        px = rng.uniform(-spawn_range_xz, spawn_range_xz)
        py = rng.uniform(-spawn_range_y, spawn_range_y)
        pz = rng.uniform(-spawn_range_xz, spawn_range_xz)
        pos = np.array([px, py, pz])

        # Rotation (Euler angles, same order as LCS)
        rot_angles = rng.uniform(0, 2 * np.pi, 3)
        R = euler_to_rotation_matrix(*rot_angles)

        # World-space centroid = R @ local_centroid + p
        center = R @ local_centroid + pos

        boxes.append(BoxObject(
            name=f"box_{i}",
            center=center,
            half_extents=np.array([half, half, half]),
            rotation=R,
            inv_mass=1.0,
        ))

    return boxes
