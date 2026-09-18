"""Session failure paths exercised against an entirely local fake SDK."""

import threading
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest
from test_device import device_pair

from wuji2_control.config import ControlConfig
from wuji2_control.errors import CleanupError, ControlError
from wuji2_control.motion import Waypoint
from wuji2_control.session import ControlSession


class Validator:
    def __init__(self, callback=None):
        self.calls, self.callback = [], callback

    def preflight(self, start, end):
        self.calls.append((tuple(start), tuple(end)))
        if self.callback:
            self.callback()


FAST = ControlConfig(
    min_move_s=0.1,
    disable_timeout_s=0.1,
    watchdog_timeout_s=0.15,
    max_run_s=3.0,
    tracking_error_duration_s=0.05,
)
POINT = Waypoint("small_move", [0.01] * 20, hold_s=0.02)


def test_success_seeds_before_enable_restores_and_closes_only_owner():
    device, sdk = device_pair()
    original = sdk.hand.settings
    validator, samples = Validator(), []
    with ControlSession(device, validator, FAST, on_sample=samples.append) as session:
        result = session.run([POINT])
        assert result["disabled"] is True
        assert result["settings_restored"] is True
    assert result["completed"] is True
    assert sdk.hand.settings == original
    events = [e[0] for e in sdk.hand.events]
    assert events.index("send") < events.index("enable") < events.index("disable")
    assert sdk.disconnected == [device.device_name]
    assert samples and samples[-1]["target"] == pytest.approx([0.01] * 20)
    assert validator.calls[0][0] == (0.0,) * 20


@pytest.mark.parametrize("failure", ["partial_enable", "partial_settings", "guard", "log"])
def test_failure_stops_and_restores_even_if_caller_catches_inside_context(failure):
    device, sdk = device_pair()
    original = sdk.hand.settings
    if failure == "partial_enable":
        sdk.hand.fail_enable = True
    if failure == "partial_settings":
        sdk.hand.fail_params_once = True

    def guard():
        if failure == "guard" and any(s == 2 for s in sdk.hand.states):
            raise RuntimeError("camera interrupted")

    def log(_):
        if failure == "log":
            raise RuntimeError("recording unavailable")

    with ControlSession(device, Validator(), FAST, guard=guard, on_sample=log) as session:
        with pytest.raises(RuntimeError):
            session.run([POINT])
        assert session.result["disabled"] is True
        assert sdk.hand.settings == original
        with pytest.raises(ControlError):
            session.run([POINT])
    assert sdk.hand.states == [1] * 20


def test_failed_stop_never_claims_disabled_or_restores_unknown_state():
    device, sdk = device_pair()
    original = sdk.hand.settings
    sdk.hand.fail_disable = sdk.hand.fail_emergency = True
    with pytest.raises(CleanupError):
        with ControlSession(device, Validator(), FAST) as session:
            session.run([POINT])
    assert session.result["disabled"] is False
    assert session.result["settings_restored"] is None
    assert sdk.hand.settings != original
    assert sdk.disconnected == [device.device_name]


def test_old_ready_diagnostics_do_not_prove_post_enable_disable():
    device, sdk = device_pair()
    sdk.hand.suppress_disable_feedback = True
    with pytest.raises(CleanupError):
        with ControlSession(device, Validator(), FAST) as session:
            session.run([POINT])
    assert session.result["disabled"] is False
    assert session.result["settings_restored"] is None


def test_hand_movement_during_preflight_prevents_enable():
    device, sdk = device_pair()

    def move_hand():
        sdk.hand.q[0] = 0.1
        sdk.hand.emit()

    with pytest.raises(ControlError, match="moved during preflight"):
        with ControlSession(device, Validator(move_hand), FAST) as session:
            session.run([POINT])
    assert not any(e[0] == "enable" for e in sdk.hand.events)
    assert session.result["settings_restored"] is True


def test_small_preflight_movement_rechecks_the_actual_start():
    device, sdk = device_pair()

    def move_hand():
        sdk.hand.q[0] = 0.02
        sdk.hand.emit()

    validator = Validator(move_hand)
    with ControlSession(device, validator, FAST) as session:
        session.run([POINT])
    assert len(validator.calls) == 2
    assert validator.calls[0][0][0] == 0.0
    assert validator.calls[1][0][0] == 0.02


def test_movement_during_final_model_check_prevents_enable():
    device, sdk = device_pair()
    calls = []

    def move_on_final_check():
        calls.append(True)
        if len(calls) == 2:
            sdk.hand.q[0] = 0.1
            sdk.hand.emit()

    with pytest.raises(ControlError, match="final preflight"):
        with ControlSession(device, Validator(move_on_final_check), FAST) as session:
            session.run([POINT])
    assert not any(event[0] == "enable" for event in sdk.hand.events)
    assert session.result["settings_restored"] is True


def test_persistent_tracking_error_stops_motion():
    device, sdk = device_pair()
    sdk.hand.follow_targets = False
    config = replace(FAST, tracking_error_rad=0.01)
    with pytest.raises(ControlError, match="tracking error"):
        with ControlSession(device, Validator(), config) as session:
            session.run([Waypoint("obstructed", [0.03] * 20, hold_s=0.1)])
    assert session.result["disabled"] is True


def test_stop_before_motion_never_enables_or_changes_settings():
    device, sdk = device_pair()
    with ControlSession(device, Validator(), FAST) as session:
        session.stop_event.set()
        with pytest.raises(ControlError):
            session.run([POINT])
    assert not any(e[0] in ("enable", "caps", "params") for e in sdk.hand.events)


@pytest.mark.parametrize("problem", ["temperature", "fault", "speed", "voltage"])
def test_runtime_diagnostic_interlocks_stop(problem):
    device, sdk = device_pair()

    def corrupt(hand, target):
        if not any(s == 2 for s in hand.states):
            return
        if problem == "temperature":
            hand.temperatures[0] = 61.0
        if problem == "fault":
            hand.errors[0] = 7
        if problem == "speed":
            hand.dq[0] = 4.1
        if problem == "voltage":
            hand.voltages[0] = 14.0

    sdk.hand.send_hook = corrupt
    with pytest.raises(ControlError):
        with ControlSession(device, Validator(), FAST) as session:
            session.run([POINT])
    # Raw diagnostics permit stopping and restoring despite the active fault.
    assert session.result["disabled"] is True
    assert session.result["settings_restored"] is True


def test_invalid_run_bound_rejected_before_settings_or_enable():
    device, sdk = device_pair()
    with ControlSession(device, Validator(), FAST) as session:
        with pytest.raises(ValueError):
            session.run([Waypoint("too_long", [0.5] * 20, hold_s=0)])
    assert not any(e[0] in ("enable", "caps", "params") for e in sdk.hand.events)


def test_watchdog_stops_blocked_publish_and_failure_remains_latched():
    device, sdk = device_pair()
    blocked, release = threading.Event(), threading.Event()

    def stall(hand, target):
        if any(s == 2 for s in hand.states):
            blocked.set()
            assert release.wait(2.0)

    sdk.hand.send_hook = stall
    outcome = []
    session = ControlSession(device, Validator(), FAST)

    def run():
        try:
            with session:
                session.run([POINT])
        except Exception as exc:
            outcome.append(exc)

    worker = threading.Thread(target=run)
    worker.start()
    try:
        assert blocked.wait(1.0)
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and not any(e[0] == "emergency" for e in sdk.hand.events):
            time.sleep(0.01)
        assert ("emergency",) in sdk.hand.events
        assert sdk.hand.states == [1] * 20
    finally:
        release.set()
        worker.join(2.0)
    assert not worker.is_alive()
    assert outcome and session.stop_event.is_set()
    assert session.result["watchdog_stop"] is True
    assert session.result["disabled"] is True
    assert session.result["completed"] is False


def test_watchdog_during_cleanup_cannot_return_success():
    device, sdk = device_pair()
    disable = sdk.hand.disable

    def delayed_disable():
        time.sleep(0.25)
        disable()

    sdk.hand.disable = delayed_disable
    with pytest.raises(ControlError):
        with ControlSession(device, Validator(), FAST) as session:
            session.run([POINT])
    assert session.result["watchdog_stop"] is True
    assert session.result["completed"] is False
    assert session.result["disabled"] is True


def test_waypoint_mutation_after_preflight_does_not_change_executed_target():
    device, sdk = device_pair()
    point = SimpleNamespace(name="mutable", q=[0.01] * 20, hold_s=0.0)

    def mutate(_):
        point.q[:] = [1.0] * 20
        point.hold_s = 100.0

    with ControlSession(device, Validator(), FAST, on_sample=mutate) as session:
        session.run([point])
    assert point.q == [1.0] * 20
    assert max(max(event[1]) for event in sdk.hand.events if event[0] == "send") <= 0.01


def test_close_failure_does_not_skip_motor_cleanup():
    device, sdk = device_pair()
    original = sdk.hand.settings
    sdk.hand.close_failures = {"command"}
    with pytest.raises(CleanupError):
        with ControlSession(device, Validator(), FAST) as session:
            session.run([POINT])
    assert sdk.hand.settings == original
    assert session.result["disabled"] is True
    assert sdk.disconnected == [device.device_name]


def test_interrupt_during_disable_still_attempts_emergency_and_restore():
    device, sdk = device_pair()
    original = sdk.hand.settings

    def interrupted_disable():
        raise KeyboardInterrupt()

    sdk.hand.disable = interrupted_disable
    with pytest.raises(CleanupError):
        with ControlSession(device, Validator(), FAST) as session:
            session.run([POINT])
    assert ("emergency",) in sdk.hand.events
    assert session.result["disabled"] is True
    assert session.result["settings_restored"] is True
    assert sdk.hand.settings == original


def test_control_config_rejects_relaxed_or_nonfinite_bounds():
    for kwargs in (
        {"max_temperature_C": 66.0},
        {"current_limit_A": 1.6},
        {"feedback_timeout_s": 1.0},
        {"speed_rad_s": float("nan")},
    ):
        with pytest.raises(ValueError):
            replace(FAST, **kwargs)
