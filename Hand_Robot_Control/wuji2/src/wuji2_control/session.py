"""A single-use, supervised position-control session with verified cleanup."""

import math
import threading
import time
from dataclasses import asdict
from typing import Any, Callable, Iterable

from .config import ControlConfig
from .device import DeviceSettings, Snapshot, Wuji2Device
from .errors import CleanupError, ControlError, FeedbackError
from .motion import Waypoint, quintic_duration, quintic_position


class ControlSession:
    """Execute validated waypoints, then disable and restore temporary settings.

    ``guard()`` must raise when an external interlock (for example recording)
    fails. ``on_sample(dict)`` receives JSON-compatible telemetry. A watchdog
    issues an independent, best-effort stop request if control or a callback
    blocks; device or transport failure can prevent stop confirmation. A session
    cannot be run twice or rearmed after an interruption.
    """

    def __init__(
        self,
        device: Wuji2Device,
        validator: Any,
        config: ControlConfig = ControlConfig(),
        guard: Callable[[], None] | None = None,
        on_sample: Callable[[dict], None] | None = None,
    ):
        self.device, self.validator, self.config = device, validator, config
        self.guard, self.on_sample = guard, on_sample
        self.stop_event = threading.Event()
        self.result: dict[str, Any] = {
            "completed": False,
            "disabled": None,
            "settings_restored": None,
            "cleanup_errors": [],
        }
        self._watchdog_shutdown = threading.Event()
        self._watchdog_thread: threading.Thread | None = None
        self._fault_lock = threading.Lock()
        self._fault: BaseException | None = None
        self._heartbeat = time.monotonic()
        self._may_be_enabled = False
        self._entered = self._ran = self._closed = False
        self._settings_attempted = self._motor_cleanup_done = False
        self._original: DeviceSettings | None = None
        self._lag_since: float | None = None
        self._enabled_at: float | None = None

    def __enter__(self) -> "ControlSession":
        if self._entered or self._closed:
            raise ControlError("A control session is single-use")
        self._entered = True
        try:
            self._external_guard()
            self.device.connect()
            deadline = time.monotonic() + self.config.disable_timeout_s
            while True:
                try:
                    snapshot = self.device.snapshot(self.config.feedback_timeout_s)
                    break
                except FeedbackError:
                    if time.monotonic() >= deadline:
                        raise
                    self._pause()
            self._check_snapshot(snapshot, expected_state=1)
            self._original = self.device.read_settings()
            self.result["original_settings"] = asdict(self._original)
            self._watchdog_thread = threading.Thread(
                target=self._watchdog, daemon=True, name="wuji2-control-watchdog"
            )
            self._watchdog_thread.start()
            return self
        except BaseException as exc:
            self._latch(exc)
            self._cleanup(close=True)
            if self.result["cleanup_errors"]:
                raise CleanupError("; ".join(self.result["cleanup_errors"])) from exc
            raise

    def _latch(self, error: BaseException) -> None:
        with self._fault_lock:
            if self._fault is None:
                self._fault = error
                self.result["error"] = str(error)
            self.result["completed"] = False
        self.stop_event.set()

    def _check_stop(self) -> None:
        if self.stop_event.is_set():
            raise ControlError(str(self._fault) if self._fault else "Session interrupted")

    def _external_guard(self) -> None:
        self._check_stop()
        if self.guard is not None:
            self.guard()
        self._check_stop()

    def _pause(self) -> None:
        # Wait from the end of work; scheduler delays never produce catch-up bursts.
        self.stop_event.wait(1 / self.config.control_hz)
        self._check_stop()

    def _watchdog(self) -> None:
        while not self._watchdog_shutdown.wait(0.02):
            if not self._may_be_enabled:
                continue
            if (
                self.stop_event.is_set()
                or time.monotonic() - self._heartbeat > self.config.watchdog_timeout_s
            ):
                self._latch(ControlError("Controller heartbeat lost or stop requested"))
                # Do not take the publisher/main-loop lock. This path must remain
                # available while send_position or a user callback is blocked.
                errors = []
                try:
                    self.device.emergency_stop()
                except Exception as exc:
                    errors.append(str(exc))
                    try:
                        self.device.disable()
                    except Exception as fallback:
                        errors.append(str(fallback))
                self.result["watchdog_stop"] = True
                if errors:
                    self.result["watchdog_errors"] = errors
                # A command return is not proof that motors are disabled.
                return

    def _check_snapshot(self, sample: Snapshot, expected_state: int | None = 2) -> None:
        if any(sample.error):
            raise ControlError(f"Device error codes: {sample.error}")
        if any(sample.position_limit) or any(sample.velocity_limit):
            raise ControlError("Device position or velocity limit active")
        if max(sample.temp_C) > self.config.max_temperature_C:
            raise ControlError("Temperature exceeds the configured session limit")
        if any(not 10.5 < value < 13.5 for value in sample.voltage_V):
            raise ControlError("Supply voltage outside session limits")
        if expected_state is not None and any(state != expected_state for state in sample.state):
            raise ControlError(f"Expected all joints in state {expected_state}")
        if max(map(abs, sample.current_A)) > self.config.current_limit_A + 0.3:
            raise ControlError("Measured current exceeds the configured session cap")
        if max(map(abs, sample.dq)) > self.config.peak_speed_rad_s:
            raise ControlError("Unexpected joint speed")

    def _tick(self, target: Iterable[float], stage: str) -> Snapshot:
        self._heartbeat = time.monotonic()
        self._external_guard()
        now = time.monotonic()
        if self._enabled_at is not None and now - self._enabled_at > self.config.max_run_s:
            raise ControlError("Enabled run exceeded its duration bound")
        sample = self.device.snapshot(self.config.feedback_timeout_s)
        self._check_snapshot(sample)
        error = max(abs(q - t) for q, t in zip(sample.q, target))
        if error > self.config.tracking_error_rad:
            self._lag_since = now if self._lag_since is None else self._lag_since
            if now - self._lag_since > self.config.tracking_error_duration_s:
                raise ControlError(f"Persistent tracking error: {error:.3f} rad")
        else:
            self._lag_since = None
        self._check_stop()
        self.device.send_position(target)
        self._check_stop()
        record = {
            **asdict(sample),
            "monotonic": now,
            "stage": stage,
            "enabled": True,
            "target": [float(q) for q in target],
            "max_error_rad": error,
        }
        self.result["last_sample"] = record
        if self.on_sample is not None:
            self.on_sample(record)
        self._check_stop()
        return sample

    def run(self, waypoints: Iterable[Waypoint]) -> dict:
        if not self._entered or self._closed or self._ran:
            raise ControlError("Run requires an entered, unused session")
        self._ran = True
        try:
            self._external_guard()
            initial = self.device.snapshot(self.config.feedback_timeout_s)
            self._check_snapshot(initial, expected_state=1)
            # Freeze even duck-typed callers' arrays before checking the path.
            points = [Waypoint(point.name, point.q, point.hold_s) for point in waypoints]
            if not points:
                raise ValueError("At least one waypoint is required")
            previous = initial.q
            expected_duration = 0.0
            for point in points:
                if (
                    not math.isfinite(point.hold_s)
                    or not 0 <= point.hold_s <= self.config.max_hold_s
                ):
                    raise ValueError("Waypoint hold exceeds the configured bound")
                self.validator.preflight(previous, point.q)
                expected_duration += (
                    quintic_duration(
                        previous, point.q, self.config.speed_rad_s, self.config.min_move_s
                    )
                    + point.hold_s
                )
                previous = point.q
            if expected_duration > self.config.max_run_s:
                raise ValueError("Planned trajectory exceeds the configured run duration")
            self.result["planned_duration_s"] = expected_duration
            self._external_guard()
            self._settings_attempted = True  # A failed RPC can still partially apply.
            self.device.write_settings(
                DeviceSettings.uniform(self.config.kp, self.config.kd, self.config.current_limit_A)
            )
            current = self.device.snapshot(self.config.feedback_timeout_s)
            self._check_snapshot(current, expected_state=1)
            if max(abs(a - b) for a, b in zip(current.q, initial.q)) > 0.05:
                raise ControlError("Hand moved during preflight; trajectory must be checked again")
            self.validator.preflight(current.q, points[0].q)
            self._external_guard()
            ready = self.device.snapshot(self.config.feedback_timeout_s)
            self._check_snapshot(ready, expected_state=1)
            if max(abs(a - b) for a, b in zip(ready.q, current.q)) > 0.001:
                raise ControlError("Hand moved during the final preflight check")
            current = ready
            self.device.send_position(current.q)
            self._check_stop()
            self._heartbeat = self._enabled_at = time.monotonic()
            self._may_be_enabled = True  # Includes partial/failed enable calls.
            self.device.enable()
            deadline = time.monotonic() + self.config.disable_timeout_s
            while True:
                self._heartbeat = time.monotonic()
                self._external_guard()
                sample = self.device.snapshot(self.config.feedback_timeout_s)
                self._check_snapshot(sample, expected_state=None)
                if all(state == 2 for state in sample.state):
                    break
                if (
                    any(state not in (1, 2) for state in sample.state)
                    or time.monotonic() >= deadline
                ):
                    raise ControlError("Enable did not reach Active on every joint")
                self.device.send_position(current.q)
                self._pause()
            previous = current.q
            for point in points:
                duration = quintic_duration(
                    previous, point.q, self.config.speed_rad_s, self.config.min_move_s
                )
                began = time.monotonic()
                while True:
                    elapsed = time.monotonic() - began
                    target = quintic_position(previous, point.q, elapsed, duration)
                    self._tick(target, point.name)
                    if elapsed >= duration:
                        break
                    self._pause()
                end = time.monotonic() + point.hold_s
                while time.monotonic() < end:
                    self._tick(point.q, point.name + "_hold")
                    self._pause()
                previous = point.q
            self.result["completed"] = True
        except BaseException as exc:
            self._latch(exc)
            raise
        finally:
            self._cleanup(close=False)
        if self.result["cleanup_errors"]:
            raise CleanupError("; ".join(self.result["cleanup_errors"]))
        self._check_stop()
        return self.result

    def _prove_disabled(self, since: float) -> bool:
        deadline = time.monotonic() + self.config.disable_timeout_s
        while time.monotonic() < deadline:
            try:
                diag = self.device.raw_diagnostics(self.config.feedback_timeout_s)
                # Ready is the only disabled state this package restores into.
                if diag.timestamp >= since and all(state == 1 for state in diag.state):
                    return True
            except Exception:
                pass
            except BaseException as exc:
                self.result["cleanup_errors"].append(
                    "Disable verification interrupted: " + repr(exc)
                )
            time.sleep(0.01)
        return False

    def _cleanup(self, *, close: bool) -> None:
        errors = self.result["cleanup_errors"]
        if not self._motor_cleanup_done:
            self._motor_cleanup_done = True
            if self._settings_attempted or self._may_be_enabled:
                since = time.monotonic()
                try:
                    self.device.disable()
                except BaseException as exc:
                    errors.append("Disable request failed: " + repr(exc))
                disabled = self._prove_disabled(since)
                if not disabled:
                    since = time.monotonic()
                    try:
                        self.device.emergency_stop()
                    except BaseException as exc:
                        errors.append("Emergency stop failed: " + repr(exc))
                    disabled = self._prove_disabled(since)
                self.result["disabled"] = disabled
                if disabled:
                    self._may_be_enabled = False
                    if self._original is not None:
                        try:
                            self.device.write_settings(self._original)
                            self.result["settings_restored"] = True
                        except BaseException as exc:
                            self.result["settings_restored"] = False
                            errors.append("Settings restoration failed: " + str(exc))
                else:
                    errors.append("Motor disable was not confirmed; settings were not restored")
                if errors:
                    self.result["completed"] = False
        if close and not self._closed:
            self._closed = True
            self._watchdog_shutdown.set()
            if self._watchdog_thread is not None:
                try:
                    self._watchdog_thread.join(timeout=self.config.disable_timeout_s)
                except BaseException as exc:
                    errors.append("Waiting for watchdog cleanup failed: " + repr(exc))
                if self._watchdog_thread.is_alive():
                    errors.append("Watchdog stop request has not returned")
            try:
                self.device.close()
            except BaseException as exc:
                errors.append("Closing device failed: " + repr(exc))
            if errors:
                self.result["completed"] = False

    def __exit__(self, exc_type: Any, exc: BaseException | None, traceback: Any) -> bool:
        if exc is not None:
            self._latch(exc)
        self._cleanup(close=True)
        if self.result["cleanup_errors"]:
            raise CleanupError("; ".join(self.result["cleanup_errors"])) from exc
        return False
