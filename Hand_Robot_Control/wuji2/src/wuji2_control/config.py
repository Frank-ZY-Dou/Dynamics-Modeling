"""Explicit connection selection and bounded, temporary control settings."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ConnectionConfig:
    serial: str | None = None
    address: str | None = None
    handedness: str = "left"

    def __post_init__(self) -> None:
        if (self.serial is None) == (self.address is None):
            raise ValueError("Specify exactly one serial or address")
        if any(
            value is not None and (not isinstance(value, str) or not value.strip())
            for value in (self.serial, self.address)
        ):
            raise ValueError("Device selector must be a nonempty string")
        if self.handedness not in ("left", "right"):
            raise ValueError("Handedness must be left or right")


@dataclass(frozen=True)
class ControlConfig:
    kp: float = 3.0
    kd: float = 0.05
    current_limit_A: float = 0.5
    speed_rad_s: float = 0.2
    control_hz: float = 100.0
    max_temperature_C: float = 60.0
    feedback_timeout_s: float = 0.3
    tracking_error_rad: float = 0.25
    tracking_error_duration_s: float = 0.5
    peak_speed_rad_s: float = 4.0
    watchdog_timeout_s: float = 0.75
    max_hold_s: float = 8.0
    max_run_s: float = 180.0
    disable_timeout_s: float = 2.0
    min_move_s: float = 2.0

    def __post_init__(self) -> None:
        bounds = {
            "kp": (3.0, 5.0),
            "kd": (0.01, 0.05),
            "current_limit_A": (0.01, 1.5),
            "speed_rad_s": (0.01, 0.3),
            "control_hz": (20.0, 200.0),
            "max_temperature_C": (1.0, 65.0),
            "feedback_timeout_s": (0.01, 0.3),
            "tracking_error_rad": (0.01, 0.25),
            "tracking_error_duration_s": (0.01, 0.5),
            "peak_speed_rad_s": (0.1, 4.0),
            "watchdog_timeout_s": (0.1, 1.5),
            "max_hold_s": (0.0, 30.0),
            "max_run_s": (1.0, 300.0),
            "disable_timeout_s": (0.1, 5.0),
            "min_move_s": (0.1, 5.0),
        }
        for name, (low, high) in bounds.items():
            value = getattr(self, name)
            if not math.isfinite(value) or not low <= value <= high:
                raise ValueError(f"{name} must be finite and within {low}..{high}")
        if self.watchdog_timeout_s <= 2 / self.control_hz:
            raise ValueError("Watchdog timeout must exceed two control periods")
