"""CLI boundaries: planning stays offline and failed recordings stay failed."""

import json
import os
import subprocess
import sys
import threading
from dataclasses import dataclass

import pytest

from wuji2_control import cli


class Model:
    def __init__(self, *args, **kwargs):
        pass

    def preflight(self, start, end):
        pass


def test_import_and_gesture_listing_do_not_load_optional_drivers():
    script = (
        "import sys; import wuji2_control; "
        "from wuji2_control.cli import main; "
        "assert main(['gestures']) == 0; "
        "assert not {'wuji_sdk','mujoco','cv2'}.intersection(sys.modules)"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert len(json.loads(result.stdout)["gestures"]) == 6


def test_run_without_execute_never_constructs_device(monkeypatch, capsys):
    monkeypatch.setattr(cli, "JointModel", Model)

    def forbidden(*args, **kwargs):
        pytest.fail("Planning must not construct a device")

    monkeypatch.setattr(cli, "Wuji2Device", forbidden)
    assert cli.main(["run", "peace", "--model", "left.xml"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["executed"] is False
    assert report["gestures"] == ["peace"]
    assert report["waypoints"]


@pytest.mark.parametrize("extra", [[], ["--camera", "0"]])
def test_execute_requires_camera_and_explicit_device(monkeypatch, tmp_path, extra):
    monkeypatch.setattr(cli, "JointModel", Model)
    assert (
        cli.main(
            [
                "run",
                "peace",
                "--model",
                "left.xml",
                "--execute",
                "--output",
                str(tmp_path),
                *extra,
            ]
        )
        == 1
    )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "args",
    [
        ["all", "peace"],
        ["unknown_pose"],
        ["peace", "--hold", "nan"],
        ["peace", "--hold", "-1"],
        ["peace", "--speed", "0"],
        ["peace", "--handedness", "right"],
    ],
)
def test_invalid_plans_rejected(monkeypatch, args):
    monkeypatch.setattr(cli, "JointModel", Model)
    assert cli.main(["check", *args, "--model", "left.xml"]) == 1


def test_inspection_only_connects_reads_and_closes(monkeypatch):
    operations = []

    @dataclass
    class State:
        q: tuple = (0.0,) * 20

    class Device:
        def __init__(self, config):
            pass

        def connect(self):
            operations.append("connect")

        def wait_for_snapshot(self):
            operations.append("read")
            return State()

        def close(self):
            operations.append("close")

    monkeypatch.setattr(cli, "Wuji2Device", Device)
    assert cli.main(["inspect", "--serial", "TEST_HAND"]) == 0
    assert operations == ["connect", "read", "close"]


def test_camera_close_failure_does_not_report_success(monkeypatch, tmp_path):
    import wuji2_control.recording as recording
    import wuji2_control.session as session_module

    monkeypatch.setattr(cli, "JointModel", Model)
    monkeypatch.setattr(cli, "Wuji2Device", lambda config: object())
    events = []

    class Camera:
        def __init__(self, *args):
            pass

        def __enter__(self):
            return self

        def check(self):
            pass

        def snapshot(self, name):
            events.append(name)

        def __exit__(self, *args):
            events.append("camera_close")
            raise RuntimeError("Could not finalize recording")

    class Session:
        def __init__(self, *args, **kwargs):
            self.stop_event = threading.Event()
            self.result = {"completed": False}

        def __enter__(self):
            return self

        def run(self, waypoints):
            self.result["completed"] = True

        def __exit__(self, *args):
            events.append("motors_disabled")
            self.result["disabled"] = True

    monkeypatch.setattr(recording, "CameraRecorder", Camera)
    monkeypatch.setattr(session_module, "ControlSession", Session)
    assert (
        cli.main(
            [
                "run",
                "peace",
                "--model",
                "left.xml",
                "--execute",
                "--serial",
                "TEST_HAND",
                "--camera",
                "0",
                "--output",
                str(tmp_path),
            ]
        )
        == 1
    )
    (report_path,) = tmp_path.glob("*/result.json")
    report = json.loads(report_path.read_text())
    assert report["completed"] is False
    assert report["disabled"] is True
    assert "finalize recording" in report["error"]
    assert events.index("motors_disabled") < events.index("camera_close")


def test_official_model_cli_check():
    model = os.environ.get("WUJI2_MODEL_PATH")
    if not model:
        pytest.skip("Set WUJI2_MODEL_PATH for the official model check")
    result = subprocess.run(
        [sys.executable, "-m", "wuji2_control", "check", "all", "--model", model],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert len(json.loads(result.stdout)["gestures"]) == 6
