"""Resting orientation of catalog assets.

An asset's authored frame is not always the pose it rests in. RoboLab's remote control is
authored standing on its 3.6 x 2.5 cm end, 16 cm tall: a pose that cannot stand with any margin
(it tips at 8.7 degrees), and every shipped scene that contains it has it lying on its back. A
layout that only sets a yaw, or a program that asks for `upright`, would otherwise stand such an
asset on end.

`simready/data/asset_rest_orientations.json` (written by `experiments/asset_rest_orientations.py`
from the catalog dimensions and the shipped scenes) lists the assets whose authored pose cannot
stand, with the authored axis that points up when they rest and the rotation that brings it to
+Z. The loaders re-express those assets in the resting frame (vertices rotated once, the pose
rotation adjusted, the world geometry unchanged), so `upright` means "resting the way RoboLab's
own scenes show it", the layouts hand the placement solver the resting footprint, and the
repair, the settle test, the export and the renders all see one body frame.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np

TABLE_PATH = Path(__file__).resolve().parents[1] / "data" / "asset_rest_orientations.json"
_TABLE: dict | None = None

_AXES = {"+x": np.array([1.0, 0.0, 0.0]), "-x": np.array([-1.0, 0.0, 0.0]),
         "+y": np.array([0.0, 1.0, 0.0]), "-y": np.array([0.0, -1.0, 0.0]),
         "+z": np.array([0.0, 0.0, 1.0]), "-z": np.array([0.0, 0.0, -1.0])}


def rotation_to_up(axis: str) -> np.ndarray:
    """The smallest rotation that brings the authored axis `axis` ('-y', '+x', ...) to +Z."""
    u = _AXES[axis]
    z = np.array([0.0, 0.0, 1.0])
    c = float(u @ z)
    if c > 1.0 - 1e-12:
        return np.eye(3)
    if c < -1.0 + 1e-12:                       # authored -Z up: turn about X by 180 degrees
        return np.diag([1.0, -1.0, -1.0])
    v = np.cross(u, z)
    s = float(np.linalg.norm(v))
    K = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    R = np.eye(3) + K + K @ K * ((1.0 - c) / (s * s))
    return R


def table() -> dict:
    global _TABLE
    if _TABLE is None:
        if TABLE_PATH.exists():
            with open(TABLE_PATH) as f:
                raw = json.load(f)
            _TABLE = {k: v for k, v in raw.get("assets", {}).items()}
        else:
            _TABLE = {}
    return _TABLE


def asset_key(path_or_name) -> str:
    """Catalog key of an asset reference: a catalog name, a USD path (its stem), or a scene child
    name with an instance suffix (`remote_control_01`)."""
    s = str(path_or_name)
    stem = os.path.splitext(os.path.basename(s))[0] if ("/" in s or s.endswith((".usd", ".usda", ".usdc"))) else s
    if stem in table():
        return stem
    base = re.sub(r"[_\-]?\d+$", "", stem)
    return base if base in table() else stem


def rest_rotation(path_or_name) -> np.ndarray | None:
    """Rotation R with R @ authored_up = +Z for an asset in the table, else None."""
    entry = table().get(asset_key(path_or_name))
    if not entry:
        return None
    return np.asarray(entry["rotation"], dtype=np.float64)


def rest_dims(dims, R: np.ndarray | None):
    """Axis-aligned extents of an asset in its resting frame (the authored extents permuted by
    the rest rotation, which is a quarter turn)."""
    d = np.asarray(dims, dtype=np.float64)
    if R is None:
        return d
    return np.abs(R) @ d
