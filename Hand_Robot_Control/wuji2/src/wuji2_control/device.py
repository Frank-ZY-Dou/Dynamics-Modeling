"""One owned Wuji SDK connection; importing this module performs no device IO."""

import importlib
import math
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable
from uuid import uuid4

from .config import ConnectionConfig
from .errors import FeedbackError, Wuji2Error

NODE_IDS = tuple(5 * f + j + 1 for f in range(5) for j in range(4))


def _values(values: Iterable[float], name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != 20 or not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain exactly 20 finite values")
    return result


@dataclass(frozen=True)
class DeviceSettings:
    kp: tuple[float, ...]
    kd: tuple[float, ...]
    current_caps_A: tuple[float, ...]

    def __post_init__(self) -> None:
        for name in ("kp", "kd", "current_caps_A"):
            values = _values(getattr(self, name), name)
            if any(value < 0 for value in values):
                raise ValueError(f"{name} cannot be negative")
            object.__setattr__(self, name, values)

    @classmethod
    def uniform(cls, kp: float, kd: float, current_limit_A: float) -> "DeviceSettings":
        return cls((kp,) * 20, (kd,) * 20, (current_limit_A,) * 20)


@dataclass(frozen=True)
class Diagnostics:
    current_A: tuple[float, ...]
    temp_C: tuple[float, ...]
    voltage_V: tuple[float, ...]
    state: tuple[int, ...]
    error: tuple[int, ...]
    position_limit: tuple[bool, ...]
    velocity_limit: tuple[bool, ...]
    timestamp: float
    age_s: float


@dataclass(frozen=True)
class Snapshot:
    q: tuple[float, ...]
    dq: tuple[float, ...]
    current_A: tuple[float, ...]
    temp_C: tuple[float, ...]
    voltage_V: tuple[float, ...]
    state: tuple[int, ...]
    error: tuple[int, ...]
    position_limit: tuple[bool, ...]
    velocity_limit: tuple[bool, ...]
    timestamp: float
    state_timestamp: float
    diagnostics_timestamp: float
    state_age_s: float
    diagnostics_age_s: float


class Wuji2Device:
    """Own a uniquely named connection; ``sdk`` may be a test double.

    Snapshot timestamps use the supplied monotonic receive clock. The SDK's
    device timestamps are not compared with the host clock: their epochs differ.
    Raw diagnostics validate transport data but deliberately retain fault flags.
    """

    def __init__(
        self,
        config: ConnectionConfig,
        *,
        sdk: Any = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.config, self._sdk, self._clock = config, sdk, clock
        self.device_name = "wuji2_control_" + uuid4().hex
        self._manager = self._hand = self._publisher = None
        self._owns_connection = False
        self._subscriptions: list[Any] = []
        self._lock = threading.Lock()
        self._frames: dict[str, tuple[Any, float]] = {}
        self._closed = False

    def connect(self) -> "Wuji2Device":
        if self._closed:
            raise Wuji2Error("A closed device cannot be reconnected")
        if self._hand is not None:
            return self
        if self._sdk is None:
            try:
                self._sdk = importlib.import_module("wuji_sdk")
            except ImportError as exc:
                raise Wuji2Error(
                    "Wuji SDK is unavailable; install wuji2-control[hardware]"
                ) from exc
        self._manager = self._sdk.SdkManager.instance()
        selector = (
            {"sn": self.config.serial} if self.config.serial else {"address": self.config.address}
        )
        try:
            self._hand = self._manager.connect(
                **selector,
                device_name=self.device_name,
                options=self._sdk.ConnectOptions(enable_bridge=False),
            )
            self._owns_connection = True
            if not isinstance(self._hand, self._sdk.WujiHand2):
                raise Wuji2Error("Selected device is not a Wuji Hand 2")
            if self._hand.handedness().get() != self.config.handedness:
                raise Wuji2Error("Selected device handedness does not match configuration")
            if self.config.serial and self._hand.serial_number != self.config.serial:
                raise Wuji2Error("Selected device serial does not match configuration")
            for kind, resource in (
                ("state", self._hand.joint_states()),
                ("diagnostics", self._hand.joint_diagnostics()),
            ):
                self._subscriptions.append(
                    resource.subscribe_with_callback(
                        lambda frame, name=kind: self._receive(name, frame)
                    )
                )
            return self
        except BaseException as exc:
            try:
                self.close()
            except Exception as cleanup:
                raise Wuji2Error(f"{exc}; connection cleanup also failed: {cleanup}") from exc
            raise

    def _receive(self, kind: str, frame: Any) -> None:
        now = self._clock()
        with self._lock:
            # Repeated SDK frame sequence numbers must not refresh stale data.
            previous = self._frames.get(kind)
            seq = getattr(getattr(frame, "header", None), "seq", None)
            if previous and seq is not None:
                old = getattr(getattr(previous[0], "header", None), "seq", None)
                if old is not None and (seq - old) % (2**32) >= 2**31:
                    return  # Out-of-order packets cannot refresh feedback age.
                if old == seq:
                    return
            self._frames[kind] = frame, now

    def _frame(self, kind: str, max_age_s: float) -> tuple[Any, float, float]:
        self._connected()
        if not math.isfinite(max_age_s) or max_age_s <= 0:
            raise ValueError("Feedback age limit must be positive and finite")
        now = self._clock()
        with self._lock:
            item = self._frames.get(kind)
        if item is None:
            raise FeedbackError(f"Missing {kind} feedback")
        frame, stamp = item
        age = now - stamp
        if not math.isfinite(now) or not math.isfinite(stamp) or not 0 <= age <= max_age_s:
            raise FeedbackError(f"Stale or future {kind} feedback")
        return frame, stamp, age

    @staticmethod
    def _ordered(frame: Any) -> list[Any]:
        joints = list(frame.joints)
        ids = [joint.nid for joint in joints]
        if (
            len(ids) != 20
            or any(type(node) is not int for node in ids)
            or len(set(ids)) != 20
            or set(ids) != set(NODE_IDS)
        ):
            raise FeedbackError("Expected exactly 20 unique known joint node IDs")
        if getattr(frame, "num_joints", 20) != 20:
            raise FeedbackError("Joint frame header count does not match payload")
        by_id = dict(zip(ids, joints))
        return [by_id[node] for node in NODE_IDS]

    def raw_diagnostics(self, max_age_s: float = 0.3) -> Diagnostics:
        frame, stamp, age = self._frame("diagnostics", max_age_s)
        joints = self._ordered(frame)
        try:
            current = _values((j.current for j in joints), "diagnostic current")
            temp = _values((j.mcu_temp_c_fb for j in joints), "temperature")
            voltage = _values((j.vbus_v_fb for j in joints), "voltage")
            states = tuple(j.status_word.ext_state for j in joints)
            errors = tuple(j.error_code_current for j in joints)
            if any(not isinstance(v, int) or v < 0 for v in states + errors):
                raise ValueError("Invalid state or error code")
            return Diagnostics(
                current,
                temp,
                voltage,
                states,
                errors,
                tuple(bool(j.status_word.position_limit_active) for j in joints),
                tuple(bool(j.status_word.velocity_limit_active) for j in joints),
                stamp,
                age,
            )
        except (TypeError, ValueError, AttributeError) as exc:
            raise FeedbackError(str(exc)) from exc

    def snapshot(self, max_age_s: float = 0.3) -> Snapshot:
        frame, stamp, age = self._frame("state", max_age_s)
        joints = self._ordered(frame)
        try:
            q = _values((j.position for j in joints), "position")
            dq = _values((j.velocity for j in joints), "velocity")
            current = _values((j.effort for j in joints), "current")
        except (TypeError, ValueError, AttributeError) as exc:
            raise FeedbackError(str(exc)) from exc
        diag = self.raw_diagnostics(max_age_s)
        return Snapshot(
            q,
            dq,
            current,
            diag.temp_C,
            diag.voltage_V,
            diag.state,
            diag.error,
            diag.position_limit,
            diag.velocity_limit,
            self._clock(),
            stamp,
            diag.timestamp,
            age,
            diag.age_s,
        )

    def wait_for_snapshot(self, timeout_s: float = 2.0, max_age_s: float = 0.3) -> Snapshot:
        """Wait briefly for the initial subscriptions; never enable or write settings."""
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("Snapshot wait timeout must be positive and finite")
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                return self.snapshot(max_age_s)
            except FeedbackError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.01)

    def _connected(self) -> Any:
        if self._hand is None or self._closed:
            raise Wuji2Error("Device is not connected")
        return self._hand

    def read_settings(self) -> DeviceSettings:
        hand = self._connected()
        params = hand.mit_params().get()
        try:
            return DeviceSettings(
                tuple(p.kp for p in params),
                tuple(p.kd for p in params),
                tuple(hand.effort_limit().get()),
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise Wuji2Error("Cannot read complete finite settings") from exc

    def write_settings(self, settings: DeviceSettings) -> None:
        hand = self._connected()
        errors = []
        # Attempt both resources even if one write fails, particularly while
        # restoring after a partial setup. Callers must keep motors disabled.
        for write in (
            lambda: hand.effort_limit().set(list(settings.current_caps_A)),
            lambda: hand.mit_params().set(list(zip(settings.kp, settings.kd))),
        ):
            try:
                write()
            except BaseException as exc:
                errors.append(repr(exc))
        try:
            actual = self.read_settings()
            for name in ("kp", "kd", "current_caps_A"):
                if any(
                    not math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-5)
                    for a, b in zip(getattr(settings, name), getattr(actual, name))
                ):
                    errors.append(f"{name} settings readback mismatch")
        except BaseException as exc:
            errors.append("Settings readback failed: " + str(exc))
        if errors:
            raise Wuji2Error("; ".join(errors))

    def send_position(self, q: Iterable[float]) -> None:
        values = _values(q, "command")
        hand = self._connected()
        if self._publisher is None:
            self._publisher = hand.joint_command().publish()
        self._publisher.send([self._sdk.JointCommand(q, 0.0, 0.0) for q in values])

    def enable(self) -> None:
        self._connected().enable()

    def disable(self) -> None:
        self._connected().disable()

    def emergency_stop(self) -> None:
        self._connected().emergency_stop()

    def close(self) -> None:
        """Close only owned resources; read-only inspection never disables motors."""
        if self._closed:
            return
        self._closed = True
        errors = []
        resources = ([self._publisher] if self._publisher is not None else []) + self._subscriptions
        for resource in resources:
            try:
                resource.close()
            except BaseException as exc:
                errors.append(repr(exc))
        if self._manager is not None and self._owns_connection:
            try:
                self._manager.disconnect(self.device_name)
            except BaseException as exc:
                errors.append(repr(exc))
        self._publisher = self._hand = None
        self._owns_connection = False
        self._subscriptions.clear()
        if errors:
            raise Wuji2Error("Resource cleanup failed: " + "; ".join(errors))
