"""The six physically demonstrated left Wuji Hand 2 Beta 2 gestures.

Targets preserve the 2026-09-17 gesture_poses.py values exactly. Positions
are in radians and SDK order: thumb, index, middle, ring, pinky, four each.
"""

from types import MappingProxyType

import numpy as np

from .motion import Waypoint

_OPEN = (0.0, 0.0, 0.0, 0.0)
_CURL = (0.78, 0.0, 1.12, 0.68)
_THUMB_OPEN = (0.0, -0.15, 0.0, 0.0)
_THUMB_TUCKED = (0.25, -0.75, 1.1, 1.1)
_THUMB_UP = (0.5, 0.22, -0.05, 0.0)
_THUMBS_UP_CURL = (0.95, 0.0, 1.12, 0.68)

_POSES = MappingProxyType(
    {
        "open_palm": _THUMB_OPEN + _OPEN * 4,
        "pointing": _THUMB_TUCKED + _OPEN + _CURL * 3,
        "peace": _THUMB_TUCKED + (0.0, -0.15, 0.0, 0.0) + (0.0, 0.12, 0.0, 0.0) + _CURL * 2,
        "thumbs_up": _THUMB_UP + _THUMBS_UP_CURL * 4,
        "i_love_you": _THUMB_OPEN + _OPEN + _CURL * 2 + _OPEN,
        "shaka": _THUMB_OPEN + _CURL * 3 + _OPEN,
    }
)


def list_gestures() -> tuple[str, ...]:
    """Return gesture names in their demonstrated order."""
    return tuple(_POSES)


def get_pose(name: str, handedness: str = "left") -> np.ndarray:
    """Return an independent mutable copy of a demonstrated left-hand pose."""
    if handedness != "left":
        raise ValueError("Only left-hand gestures have been validated")
    if not isinstance(name, str) or name not in _POSES:
        raise ValueError(f"Unknown gesture {name!r}; choose from {', '.join(_POSES)}")
    return np.array(_POSES[name], dtype=float, copy=True)


def gesture_sequence(names, hold_s=4) -> list[Waypoint]:
    """Start open, present each requested gesture, and return open after each."""
    if isinstance(names, (str, bytes)):
        raise ValueError("names must be a sequence of gesture names")
    requested = tuple(names)
    poses = [(name, get_pose(name)) for name in requested]
    result = [Waypoint("open_palm", get_pose("open_palm"), hold_s)]
    for name, pose in poses:
        result.append(Waypoint(name, pose, hold_s))
        result.append(Waypoint("open_palm", get_pose("open_palm"), hold_s))
    return result
