"""Offline transport tests: no vendor SDK or connected hardware required."""

import threading
import time
from types import SimpleNamespace as NS

import pytest

from wuji2_control.config import ConnectionConfig
from wuji2_control.device import NODE_IDS, DeviceSettings, Wuji2Device
from wuji2_control.errors import FeedbackError, Wuji2Error


class Resource:
    def __init__(self, hand, kind):
        self.hand, self.kind = hand, kind

    def subscribe_with_callback(self, callback):
        self.hand.callbacks[self.kind] = callback
        self.hand.emit()
        return self

    def get(self):
        if self.kind == "side":
            return self.hand.side
        if self.kind == "caps":
            return list(self.hand.settings.current_caps_A)
        return [NS(kp=kp, kd=kd) for kp, kd in zip(self.hand.settings.kp, self.hand.settings.kd)]

    def set(self, value):
        self.hand.events.append((self.kind, value))
        previous = self.hand.settings
        if self.kind == "caps":
            self.hand.settings = DeviceSettings(previous.kp, previous.kd, tuple(value))
        else:
            if self.hand.fail_params_once:
                self.hand.fail_params_once = False
                raise RuntimeError("partial settings failure")
            self.hand.settings = DeviceSettings(
                tuple(p[0] for p in value), tuple(p[1] for p in value), previous.current_caps_A
            )

    def publish(self):
        return self

    def send(self, values):
        target = tuple(v.position for v in values)
        self.hand.events.append(("send", target))
        if self.hand.send_hook:
            self.hand.send_hook(self.hand, target)
        if any(state == 2 for state in self.hand.states) and self.hand.follow_targets:
            self.hand.q = list(target)
        self.hand.emit()

    def close(self):
        self.hand.events.append(("close", self.kind))
        if self.kind in self.hand.close_failures:
            raise RuntimeError(self.kind + " close failed")


class FakeHand:
    def __init__(self):
        self.serial_number, self.side = "test-device", "left"
        self.events, self.callbacks = [], {}
        self.settings = DeviceSettings.uniform(1.5, 0.1, 1.5)
        self.q, self.dq = [0.0] * 20, [0.0] * 20
        self.states, self.errors = [1] * 20, [0] * 20
        self.temperatures, self.voltages = [30.0] * 20, [12.0] * 20
        self.current = [0.0] * 20
        self.seq = 0
        self.follow_targets = True
        self.fail_params_once = self.fail_enable = self.fail_disable = self.fail_emergency = False
        self.suppress_disable_feedback = False
        self.send_hook = None
        self.close_failures = set()

    def frame(self, kind):
        joints = []
        for i, nid in enumerate(NODE_IDS):
            if kind == "state":
                joints.append(
                    NS(nid=nid, position=self.q[i], velocity=self.dq[i], effort=self.current[i])
                )
            else:
                status = NS(
                    ext_state=self.states[i],
                    position_limit_active=False,
                    velocity_limit_active=False,
                )
                joints.append(
                    NS(
                        nid=nid,
                        status_word=status,
                        current=self.current[i],
                        mcu_temp_c_fb=self.temperatures[i],
                        vbus_v_fb=self.voltages[i],
                        error_code_current=self.errors[i],
                    )
                )
        return NS(joints=joints, num_joints=20, header=NS(seq=self.seq))

    def emit(self):
        self.seq += 1
        for kind, callback in self.callbacks.items():
            callback(self.frame(kind))

    def handedness(self):
        return Resource(self, "side")

    def joint_states(self):
        return Resource(self, "state")

    def joint_diagnostics(self):
        return Resource(self, "diagnostics")

    def joint_command(self):
        return Resource(self, "command")

    def mit_params(self):
        return Resource(self, "params")

    def effort_limit(self):
        return Resource(self, "caps")

    def enable(self):
        self.events.append(("enable",))
        self.states = [2] * 10 + [1 if self.fail_enable else 2] * 10
        self.emit()
        if self.fail_enable:
            raise RuntimeError("partial enable failure")

    def disable(self):
        self.events.append(("disable",))
        if self.fail_disable:
            raise RuntimeError("disable transport failure")
        self.states = [1] * 20
        if not self.suppress_disable_feedback:
            self.emit()

    def emergency_stop(self):
        self.events.append(("emergency",))
        if self.fail_emergency:
            raise RuntimeError("emergency transport failure")
        self.states = [1] * 20
        if not self.suppress_disable_feedback:
            self.emit()


class FakeSDK:
    WujiHand2 = FakeHand
    ConnectOptions = staticmethod(lambda **kwargs: NS(**kwargs))
    JointCommand = staticmethod(lambda q, dq, effort: NS(position=q, velocity=dq, effort=effort))

    def __init__(self, hand=None):
        self.hand = hand or FakeHand()
        self.calls, self.disconnected = [], []
        self.SdkManager = NS(instance=lambda: self)

    def connect(self, **kwargs):
        self.calls.append(kwargs)
        return self.hand

    def disconnect(self, name):
        self.disconnected.append(name)

    def disconnect_all(self):
        raise AssertionError("Must not disconnect another application")


def device_pair(*, clock=time.monotonic):
    sdk = FakeSDK()
    device = Wuji2Device(ConnectionConfig(serial="test-device"), sdk=sdk, clock=clock)
    return device, sdk


def test_explicit_selection_and_lazy_connection():
    with pytest.raises(ValueError):
        ConnectionConfig()
    with pytest.raises(ValueError):
        ConnectionConfig(serial="a", address="b")
    device, sdk = device_pair()
    assert not sdk.calls
    device.connect()
    assert sdk.calls[0]["sn"] == "test-device"
    assert sdk.calls[0]["options"].enable_bridge is False
    device.close()
    assert sdk.disconnected == [device.device_name]
    assert not any(e[0] in ("enable", "disable", "emergency") for e in sdk.hand.events)


@pytest.mark.parametrize("mismatch", ["side", "serial", "type"])
def test_identity_mismatch_closes_owned_connection(mismatch):
    device, sdk = device_pair()
    if mismatch == "side":
        sdk.hand.side = "right"
    if mismatch == "serial":
        sdk.hand.serial_number = "other"
    if mismatch == "type":
        sdk.hand = object()
    with pytest.raises(Wuji2Error):
        device.connect()
    assert sdk.disconnected == [device.device_name]


def test_joint_mapping_is_by_id_not_stream_order():
    device, sdk = device_pair()
    device.connect()
    sdk.hand.q = [i / 100 for i in range(20)]
    frame = sdk.hand.frame("state")
    frame.header.seq += 1
    frame.joints.reverse()
    device._receive("state", frame)
    assert device.snapshot().q == tuple(sdk.hand.q)


@pytest.mark.parametrize("kind", ["state", "diagnostics"])
@pytest.mark.parametrize("corruption", ["missing", "duplicate", "unknown", "nonfinite"])
def test_corrupt_feedback_rejected(kind, corruption):
    device, sdk = device_pair()
    device.connect()
    frame = sdk.hand.frame(kind)
    frame.header.seq += 1
    if corruption == "missing":
        frame.joints.pop()
    if corruption == "duplicate":
        frame.joints[-1].nid = frame.joints[0].nid
    if corruption == "unknown":
        frame.joints[-1].nid = 999
    if corruption == "nonfinite":
        setattr(frame.joints[0], "position" if kind == "state" else "mcu_temp_c_fb", float("nan"))
    device._receive(kind, frame)
    with pytest.raises(FeedbackError):
        device.snapshot()


def test_stale_future_and_repeated_frames_rejected():
    now = [10.0]
    device, sdk = device_pair(clock=lambda: now[0])
    device.connect()
    now[0] = 10.31
    device._receive("state", sdk.hand.frame("state"))  # Duplicate sequence does not refresh.
    with pytest.raises(FeedbackError):
        device.snapshot()
    now[0] = 9.9
    with pytest.raises(FeedbackError):
        device.snapshot()


def test_sequence_reordering_and_32_bit_wrap():
    now = [10.0]
    device, sdk = device_pair(clock=lambda: now[0])
    device.connect()
    sdk.hand.seq = 100
    device._receive("state", sdk.hand.frame("state"))
    now[0] = 10.31
    sdk.hand.seq = 99
    device._receive("state", sdk.hand.frame("state"))
    with pytest.raises(FeedbackError):
        device.snapshot()
    # A fresh connection can cross the unsigned 32-bit counter boundary.
    device, sdk = device_pair(clock=lambda: now[0])
    sdk.hand.seq = 2**32 - 3
    device.connect()
    sdk.hand.seq = 0
    sdk.hand.q[0] = 0.123
    device._receive("state", sdk.hand.frame("state"))
    assert device.snapshot().q[0] == 0.123


def test_raw_diagnostics_remain_available_with_fault_and_stale_state():
    now = [1.0]
    device, sdk = device_pair(clock=lambda: now[0])
    device.connect()
    now[0] = 1.4
    sdk.hand.errors[0] = 7
    frame = sdk.hand.frame("diagnostics")
    frame.header.seq += 1
    device._receive("diagnostics", frame)
    assert device.raw_diagnostics().error[0] == 7
    with pytest.raises(FeedbackError):
        device.snapshot()


def test_settings_round_trip_and_invalid_command():
    device, sdk = device_pair()
    device.connect()
    original = device.read_settings()
    device.write_settings(DeviceSettings.uniform(3.0, 0.05, 0.5))
    assert device.read_settings().current_caps_A == (0.5,) * 20
    device.write_settings(original)
    assert device.read_settings() == original
    for q in ([0.0] * 19, [float("inf")] * 20):
        with pytest.raises(ValueError):
            device.send_position(q)
    assert not any(e[0] == "send" for e in sdk.hand.events)


def test_all_close_actions_attempted_after_individual_failure():
    device, sdk = device_pair()
    device.connect()
    device.send_position([0.0] * 20)
    sdk.hand.close_failures = {"command", "state"}
    with pytest.raises(Wuji2Error):
        device.close()
    assert ("close", "diagnostics") in sdk.hand.events
    assert sdk.disconnected == [device.device_name]


def test_wait_for_initial_snapshot_is_read_only():
    device, sdk = device_pair()
    device.connect()
    with device._lock:
        device._frames.clear()
    timer = threading.Timer(0.02, sdk.hand.emit)
    timer.start()
    try:
        assert device.wait_for_snapshot(timeout_s=0.2).q == (0.0,) * 20
        device.read_settings()
    finally:
        timer.join()
        device.close()
    assert not any(
        e[0] in ("caps", "params", "send", "enable", "disable", "emergency")
        for e in sdk.hand.events
    )


def test_partial_subscription_failure_releases_first_subscription_and_owner():
    device, sdk = device_pair()

    def fail_subscription(_):
        raise RuntimeError("diagnostic subscribe failed")

    sdk.hand.joint_diagnostics = lambda: NS(subscribe_with_callback=fail_subscription)
    with pytest.raises(RuntimeError, match="subscribe failed"):
        device.connect()
    assert ("close", "state") in sdk.hand.events
    assert sdk.disconnected == [device.device_name]


def test_failed_connect_does_not_disconnect_an_unowned_connection():
    device, sdk = device_pair()

    def fail_connect(**kwargs):
        raise RuntimeError("connect failed")

    sdk.connect = fail_connect
    with pytest.raises(RuntimeError, match="connect failed"):
        device.connect()
    assert not sdk.disconnected


def test_missing_sdk_has_install_hint(monkeypatch):
    device = Wuji2Device(ConnectionConfig(serial="test-device"))

    def absent(_):
        raise ImportError("missing")

    monkeypatch.setattr("wuji2_control.device.importlib.import_module", absent)
    with pytest.raises(Wuji2Error, match=r"wuji2-control\[hardware\]"):
        device.connect()


def test_settings_readback_mismatch_is_reported():
    device, sdk = device_pair()
    device.connect()
    resource = Resource(sdk.hand, "caps")
    resource.get = lambda: [0.9] * 20
    sdk.hand.effort_limit = lambda: resource
    with pytest.raises(Wuji2Error, match="readback mismatch"):
        device.write_settings(DeviceSettings.uniform(3.0, 0.05, 0.5))


def test_closed_device_cannot_supply_recent_cached_snapshot():
    device, sdk = device_pair()
    device.connect()
    device.close()
    with pytest.raises(Wuji2Error, match="not connected"):
        device.snapshot()
