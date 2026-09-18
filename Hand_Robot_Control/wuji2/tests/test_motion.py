from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from wuji2_control.motion import Waypoint, quintic_duration, quintic_position


def test_waypoint_owns_immutable_position_data():
    source = np.arange(20, dtype=float)
    waypoint = Waypoint("pose", source, 0)
    source[:] = -1
    np.testing.assert_array_equal(waypoint.q, np.arange(20))
    with pytest.raises(ValueError):
        waypoint.q[0] = 3
    with pytest.raises(ValueError):
        waypoint.q.setflags(write=True)
    reshaped_view = waypoint.q
    reshaped_view.shape = (5, 4)
    assert waypoint.q.shape == (20,)
    with pytest.raises(FrozenInstanceError):
        waypoint.hold_s = 2
    assert waypoint == Waypoint("pose", np.arange(20), 0)
    assert hash(waypoint) == hash(Waypoint("pose", np.arange(20), 0))


@pytest.mark.parametrize("hold", [-1, float("nan"), float("inf"), True])
def test_waypoint_rejects_invalid_hold(hold):
    with pytest.raises(ValueError):
        Waypoint("pose", np.zeros(20), hold)


@pytest.mark.parametrize(
    "q",
    [
        np.zeros(19),
        np.zeros((5, 4)),
        [float("nan")] * 20,
        [float("inf")] * 20,
        np.ones(20, dtype=complex) * 1j,
    ],
)
def test_motion_rejects_invalid_vectors(q):
    with pytest.raises(ValueError):
        Waypoint("pose", q, 1)
    with pytest.raises(ValueError):
        quintic_duration(q, np.zeros(20), 0.2)
    with pytest.raises(ValueError):
        quintic_position(np.zeros(20), q, 1, 2)


def test_quintic_peak_velocity_respects_limit_for_every_joint():
    start = np.linspace(-0.4, 0.5, 20)
    end = start[::-1] + 0.8
    speed = 0.23
    duration = quintic_duration(start, end, speed)
    times = np.linspace(0, duration, 10001)
    positions = np.array([quintic_position(start, end, t, duration) for t in times])
    velocities = np.diff(positions, axis=0) / (times[1] - times[0])
    assert np.max(np.abs(velocities)) <= speed * (1 + 1e-8)
    assert np.max(np.abs(velocities)) == pytest.approx(speed, rel=1e-6)
    # The endpoints have zero velocity and acceleration, including a hold.
    h = duration * 1e-4
    for t in [0.0, duration]:
        before = quintic_position(start, end, t - h, duration)
        center = quintic_position(start, end, t, duration)
        after = quintic_position(start, end, t + h, duration)
        assert np.max(np.abs((after - before) / (2 * h))) < 1e-6
        assert np.max(np.abs((after - 2 * center + before) / h**2)) < 1e-4


def test_quintic_exact_endpoints_and_stationary_motion():
    start = np.linspace(-0.3, 0.7, 20)
    end = start + 0.5
    assert quintic_duration(start, start, 0.2) == 2
    assert quintic_duration(start, end, 0.2) == pytest.approx(1.875 * 0.5 / 0.2)
    for elapsed in [-1, 0]:
        result = quintic_position(start, end, elapsed, 3)
        np.testing.assert_array_equal(result, start)
        result[:] = 0
        assert not np.shares_memory(result, start)
    for elapsed in [3, 4]:
        np.testing.assert_array_equal(quintic_position(start, end, elapsed, 3), end)
    np.testing.assert_allclose(quintic_position(start, end, 1.5, 3), (start + end) / 2)
    np.testing.assert_array_equal(quintic_position(start, start, 1, 2), start)


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True])
def test_invalid_speed_and_duration_are_rejected(value):
    q = np.zeros(20)
    with pytest.raises(ValueError):
        quintic_duration(q, q, value)
    with pytest.raises(ValueError):
        quintic_duration(q, q, 0.2, min_duration=value)
    with pytest.raises(ValueError):
        quintic_position(q, q, 1, value)


def test_nonfinite_elapsed_is_rejected():
    with pytest.raises(ValueError):
        quintic_position(np.zeros(20), np.ones(20), float("nan"), 2)
