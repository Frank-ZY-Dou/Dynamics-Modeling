"""Validated joint vectors and smooth, speed-limited motion primitives."""

import math
from dataclasses import dataclass
from numbers import Real

import numpy as np


def _joint_vector(value, name: str = "q") -> np.ndarray:
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must contain real joint positions")
    try:
        vector = np.array(raw, dtype=float, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite vector of 20 joint positions") from exc
    if vector.shape != (20,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must be a finite vector of 20 joint positions")
    return vector


def _finite_scalar(value, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        qualifier = "positive and finite" if positive else "finite"
        raise ValueError(f"{name} must be {qualifier}")
    return result


def _difference(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        difference = end - start
    if not np.isfinite(difference).all():
        raise ValueError("Joint displacement exceeds finite floating-point range")
    return difference


@dataclass(frozen=True, slots=True, eq=False, init=False, repr=False)
class Waypoint:
    """A named pose and nonnegative hold time, with immutable position data."""

    name: str
    _q_bytes: bytes
    hold_s: float

    def __init__(self, name: str, q, hold_s: float):
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Waypoint name must be a nonempty string")
        vector = _joint_vector(q)
        hold = _finite_scalar(hold_s, "hold_s")
        if hold < 0:
            raise ValueError("hold_s must be nonnegative")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "_q_bytes", vector.tobytes())
        object.__setattr__(self, "hold_s", hold)

    @property
    def q(self) -> np.ndarray:
        # A fresh view also prevents array shape/dtype edits changing the pose.
        return np.frombuffer(self._q_bytes, dtype=np.float64)

    def __repr__(self) -> str:
        return f"Waypoint(name={self.name!r}, q={self.q.tolist()!r}, hold_s={self.hold_s!r})"

    def __eq__(self, other):
        if not isinstance(other, Waypoint):
            return NotImplemented
        return (
            self.name == other.name
            and self.hold_s == other.hold_s
            and np.array_equal(self.q, other.q)
        )

    def __hash__(self) -> int:
        return hash((self.name, tuple(self.q), self.hold_s))


def quintic_duration(start, end, speed, min_duration=2) -> float:
    """Return a duration bounding every joint's peak speed in rad/s.

    The smoothstep polynomial has maximum derivative 1.875, so a distance
    divided by speed alone would exceed the requested speed limit.
    """
    start = _joint_vector(start, "start")
    end = _joint_vector(end, "end")
    speed = _finite_scalar(speed, "speed", positive=True)
    minimum = _finite_scalar(min_duration, "min_duration", positive=True)
    distance = float(np.max(np.abs(_difference(start, end))))
    duration = max(minimum, 1.875 * (distance / speed))
    if not math.isfinite(duration):
        raise ValueError("Requested movement duration is not finite")
    return duration


def quintic_position(start, end, elapsed, duration) -> np.ndarray:
    """Evaluate a C2-continuous transition, held constant outside its interval."""
    start = _joint_vector(start, "start")
    end = _joint_vector(end, "end")
    elapsed = _finite_scalar(elapsed, "elapsed")
    duration = _finite_scalar(duration, "duration", positive=True)
    if elapsed <= 0:
        return start
    if elapsed >= duration:
        return end
    fraction = elapsed / duration
    blend = fraction**3 * (10 + fraction * (-15 + 6 * fraction))
    return start + blend * _difference(start, end)
