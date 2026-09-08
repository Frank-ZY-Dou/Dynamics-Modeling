"""A faithful replica of RoboCasa's placement validity test (robosuite-style UniformRandomSampler
as vendored in robocasa/utils/placement_samplers.py + object_utils.py), for MJCF objects:

  * an object's footprint is its `reg_bbox` region box (centre, half size) from model.xml;
  * `_sample_x/_sample_y`: uniform in the region range shrunk by min(w, d) / 2 (w, d = full
    bbox extents), `ensure_object_boundary_in_range=True`;
  * z: the bbox bottom sits on the reference top (object_z = top - bottom_offset);
  * yaw: uniform in [0, 2 pi) about z (`rotation=None`, `rotation_axis="z"`);
  * `obj_in_region`: all 8 rotated bbox corners inside the (unshrunk) region rectangle (xy only);
  * `objs_intersect`: separating-axis test between the two rotated boxes on their 6 face normals;
  * 5000 attempts per object, then PlacementError ("Cannot place all objects").

robosuite itself is not installed here; this mirrors the vendored source line by line
(placement_samplers.py:225-262, 303-470; object_utils.py:249-290, 507-540).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import simready  # noqa: E402,F401
from simready.io.mjcf_io import region_bbox  # noqa: E402


class PlacementError(Exception):
    pass


def bbox_points(center, half, trans, R):
    offs = [center + half * np.array(s) for s in ([-1, -1, -1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1], [1, 1, 1], [-1, 1, 1], [1, -1, 1], [1, 1, -1])]
    return [R @ p + trans for p in offs]


def in_region(points, p0, px, py):
    u = px - p0; v = py - p0
    for pt in points:
        if not (np.dot(u, p0) <= np.dot(u, pt) <= np.dot(u, px)):
            return False
        if not (np.dot(v, p0) <= np.dot(v, pt) <= np.dot(v, py)):
            return False
    return True


def boxes_intersect(pa, pb):
    normals = [pa[1] - pa[0], pa[2] - pa[0], pa[3] - pa[0], pb[1] - pb[0], pb[2] - pb[0], pb[3] - pb[0]]
    for nrm in normals:
        nrm = np.asarray(nrm) / np.linalg.norm(nrm)
        a = [np.dot(p, nrm) for p in pa]; b = [np.dot(p, nrm) for p in pb]
        if min(b) > max(a) or min(a) > max(b):
            return False
    return True


def yaw_R(t):
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class ObjectSpec:
    def __init__(self, xml):
        self.xml = str(xml)
        rb = region_bbox(xml)
        if rb is None:
            raise ValueError(f"{xml}: no reg_bbox")
        self.center, self.half = rb
        self.size = 2.0 * self.half                       # (w, d, h) as px - p0, py - p0, pz - p0
        self.bottom_offset = float(self.center[2] - self.half[2])
        self.horizontal_radius = float(np.linalg.norm(self.half[:2]))


def sample_layout(rng, objs, x_range, y_range, base=(0.0, 0.0, 0.0), attempts=5000, partial=False):
    """Place `objs` (ObjectSpec list) in the region; returns [(x, y, z, yaw)] or raises PlacementError
    (with partial=True returns the placements made before the failure plus None)."""
    base = np.asarray(base, dtype=np.float64)
    region = [np.array([x_range[0], y_range[0], 0.0]) + base, np.array([x_range[1], y_range[0], 0.0]) + base,
              np.array([x_range[0], y_range[1], 0.0]) + base]
    placed = []      # (pos, R, spec)
    out = []
    for spec in objs:
        buf = min(spec.size[0], spec.size[1]) / 2
        xmin, xmax = x_range[0] + buf, x_range[1] - buf
        ymin, ymax = y_range[0] + buf, y_range[1] - buf
        if xmin > xmax or ymin > ymax:
            if partial:
                return out, False
            raise PlacementError("Invalid range for placement initializer")
        ok = False
        for _ in range(attempts):
            x = float(rng.uniform(xmin, xmax)) + base[0]
            y = float(rng.uniform(ymin, ymax)) + base[1]
            z = float(base[2] - spec.bottom_offset)
            yaw = float(rng.uniform(0.0, 2.0 * math.pi))
            R = yaw_R(yaw); pos = np.array([x, y, z])
            pts = bbox_points(spec.center, spec.half, pos, R)
            if not in_region(pts, region[0], region[1], region[2]):
                continue
            if any(boxes_intersect(pts, bbox_points(o.center, o.half, p, r)) for p, r, o in placed):
                continue
            placed.append((pos, R, spec)); out.append((x, y, z, yaw)); ok = True
            break
        if not ok:
            if partial:
                return out, False
            raise PlacementError("Cannot place all objects")
    return (out, True) if partial else out
