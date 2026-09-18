import importlib
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from wuji2_control.gestures import get_pose, list_gestures
from wuji2_control.model import SDK_JOINT_NAMES, JointModel


def write_fixture(
    tmp_path,
    *,
    order=SDK_JOINT_NAMES,
    collision=False,
    joint_type="hinge",
    free_base=False,
    first_range="-2 2",
    limited=True,
):
    root = ET.Element("mujoco", model="fixture-left")
    ET.SubElement(root, "compiler", angle="radian")
    world = ET.SubElement(root, "worldbody")
    parent = world
    if free_base:
        parent = ET.SubElement(world, "body", name="free_base")
        ET.SubElement(parent, "freejoint")
        ET.SubElement(parent, "geom", type="sphere", size=".01", pos="-2 0 0")
    if collision:
        ET.SubElement(world, "geom", name="obstacle", type="sphere", size=".015", pos=".15 0 0")
    for i, name in enumerate(order):
        special = name == SDK_JOINT_NAMES[0]
        body = ET.SubElement(
            parent, "body", name=f"body_{i}", pos="0 0 0" if special else f"0 {i + 1} 0"
        )
        ET.SubElement(
            body,
            "joint",
            name=name,
            type=joint_type if special else "hinge",
            axis="0 0 1",
            range=first_range if special else "-2 2",
            limited="true" if limited else "false",
        )
        ET.SubElement(
            body,
            "geom",
            type="sphere",
            size=".015",
            pos=".15 0 0" if special and collision else "0 0 0",
        )
    path = tmp_path / "fixture.xml"
    ET.ElementTree(root).write(path)
    return path


@pytest.fixture
def mujoco_available():
    return pytest.importorskip("mujoco")


def test_importing_model_does_not_import_mujoco(monkeypatch):
    monkeypatch.setitem(sys.modules, "mujoco", None)
    module = importlib.reload(importlib.import_module("wuji2_control.model"))
    assert module.JointModel is not None


def test_explicit_model_path_is_required(tmp_path):
    with pytest.raises(TypeError):
        JointModel()
    with pytest.raises(FileNotFoundError):
        JointModel(tmp_path / "not-present.xml")
    with pytest.raises(ValueError, match="left"):
        JointModel(tmp_path / "not-present.xml", handedness="right")


def test_named_mapping_accepts_reordered_storage(tmp_path, mujoco_available):
    model = JointModel(write_fixture(tmp_path, order=SDK_JOINT_NAMES[::-1], first_range="-.4 .4"))
    q = np.zeros(20)
    q[-1] = 1.2
    model.validate(q)
    np.testing.assert_array_equal(model.joint_limits[0], [-0.4, 0.4])
    exposed = model.joint_limits
    exposed[:] = 0
    assert model.joint_limits[0, 1] == 0.4
    q[0] = 0.36
    with pytest.raises(ValueError, match="l_thumb_cmc_flex"):
        model.validate(q)


def test_midtrajectory_collision_is_rejected_even_with_clear_endpoints(tmp_path, mujoco_available):
    model = JointModel(write_fixture(tmp_path, collision=True, order=SDK_JOINT_NAMES[::-1]))
    start = np.zeros(20)
    end = start.copy()
    start[0], end[0] = -0.8, 0.8
    model.validate(start)
    model.validate(end)
    with pytest.raises(ValueError, match="Trajectory sample.*Self-collision"):
        model.preflight(start, end)


@pytest.mark.parametrize(
    "change", ["missing", "extra", "right", "unnamed", "slide", "free", "unlimited"]
)
def test_incompatible_models_fail_before_use(tmp_path, mujoco_available, change):
    options = {}
    if change == "missing":
        options["order"] = SDK_JOINT_NAMES[:-1]
    elif change == "extra":
        options["order"] = SDK_JOINT_NAMES + ("extra_joint",)
    elif change == "right":
        options["order"] = tuple("r_" + name[2:] for name in SDK_JOINT_NAMES)
    elif change == "unnamed":
        options["order"] = SDK_JOINT_NAMES[:-1] + ("unrecognized_joint",)
    elif change == "slide":
        options["joint_type"] = "slide"
    elif change == "free":
        options["free_base"] = True
    elif change == "unlimited":
        options["limited"] = False
    with pytest.raises(ValueError):
        JointModel(write_fixture(tmp_path, **options))


def test_vector_and_configuration_validation(tmp_path, mujoco_available):
    path = write_fixture(tmp_path)
    model = JointModel(path)
    for q in [np.zeros(19), np.zeros((5, 4)), [float("nan")] * 20, [float("inf")] * 20]:
        with pytest.raises(ValueError):
            model.validate(q)
    for kwargs in [
        {"margin": -1},
        {"margin": float("nan")},
        {"margin": 2},
        {"penetration_tolerance": -1},
        {"penetration_tolerance": float("inf")},
        {"samples": 2},
        {"samples": 3.5},
    ]:
        with pytest.raises(ValueError):
            JointModel(path, **kwargs)
    q = np.zeros(20)
    q[0] = 1.951
    with pytest.raises(ValueError, match="margin-adjusted"):
        model.preflight(np.zeros(20), q)


def test_official_model_all_demonstrated_gesture_paths(mujoco_available):
    configured = os.environ.get("WUJI2_MODEL_PATH")
    if not configured:
        pytest.skip("Set WUJI2_MODEL_PATH to an existing official left MJCF")
    model = JointModel(Path(configured))
    opened = get_pose("open_palm")
    for name in list_gestures():
        pose = get_pose(name)
        model.validate(pose)
        model.preflight(opened, pose)
        model.preflight(pose, opened)
