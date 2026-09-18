import numpy as np
import pytest

from wuji2_control.gestures import gesture_sequence, get_pose, list_gestures

# Archived targets from gesture_poses.py, demonstrated in 20260917_184038_all.
DEMONSTRATED = {
    "open_palm": [0, -0.15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "pointing": [
        0.25,
        -0.75,
        1.1,
        1.1,
        0,
        0,
        0,
        0,
        0.78,
        0,
        1.12,
        0.68,
        0.78,
        0,
        1.12,
        0.68,
        0.78,
        0,
        1.12,
        0.68,
    ],
    "peace": [
        0.25,
        -0.75,
        1.1,
        1.1,
        0,
        -0.15,
        0,
        0,
        0,
        0.12,
        0,
        0,
        0.78,
        0,
        1.12,
        0.68,
        0.78,
        0,
        1.12,
        0.68,
    ],
    "thumbs_up": [
        0.5,
        0.22,
        -0.05,
        0,
        0.95,
        0,
        1.12,
        0.68,
        0.95,
        0,
        1.12,
        0.68,
        0.95,
        0,
        1.12,
        0.68,
        0.95,
        0,
        1.12,
        0.68,
    ],
    "i_love_you": [
        0,
        -0.15,
        0,
        0,
        0,
        0,
        0,
        0,
        0.78,
        0,
        1.12,
        0.68,
        0.78,
        0,
        1.12,
        0.68,
        0,
        0,
        0,
        0,
    ],
    "shaka": [
        0,
        -0.15,
        0,
        0,
        0.78,
        0,
        1.12,
        0.68,
        0.78,
        0,
        1.12,
        0.68,
        0.78,
        0,
        1.12,
        0.68,
        0,
        0,
        0,
        0,
    ],
}


def test_all_six_targets_are_preserved_exactly():
    assert list_gestures() == tuple(DEMONSTRATED)
    for name, expected in DEMONSTRATED.items():
        np.testing.assert_array_equal(get_pose(name), expected)


def test_pose_copies_cannot_change_catalog_or_other_waypoints():
    pose = get_pose("pointing")
    pose[:] = 100
    np.testing.assert_array_equal(get_pose("pointing"), DEMONSTRATED["pointing"])
    waypoints = gesture_sequence(["pointing", "peace"], hold_s=1.5)
    assert [step.name for step in waypoints] == [
        "open_palm",
        "pointing",
        "open_palm",
        "peace",
        "open_palm",
    ]
    assert all(step.hold_s == 1.5 for step in waypoints)
    assert not np.shares_memory(waypoints[0].q, waypoints[2].q)
    with pytest.raises(ValueError):
        waypoints[1].q[0] = 1


def test_empty_sequence_still_starts_open():
    sequence = gesture_sequence([])
    assert len(sequence) == 1
    assert sequence[0].name == "open_palm"
    assert sequence[0].hold_s == 4


@pytest.mark.parametrize("name", ["unknown", "all", None])
def test_unknown_names_are_rejected(name):
    with pytest.raises(ValueError, match="Unknown gesture"):
        get_pose(name)


def test_unvalidated_right_hand_and_bad_sequence_are_rejected():
    with pytest.raises(ValueError, match="left"):
        get_pose("open_palm", handedness="right")
    with pytest.raises(ValueError, match="sequence"):
        gesture_sequence("peace")
    with pytest.raises(ValueError):
        gesture_sequence(["peace"], hold_s=float("nan"))
