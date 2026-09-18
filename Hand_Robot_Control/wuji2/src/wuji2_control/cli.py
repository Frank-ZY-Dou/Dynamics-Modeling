"""Command-line discovery, inspection, gesture control, and video export."""

from __future__ import annotations

import argparse
import json
import signal
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from . import __version__
from .config import ConnectionConfig, ControlConfig
from .device import Wuji2Device
from .gestures import gesture_sequence, get_pose, list_gestures
from .model import JointModel
from .motion import quintic_duration


def _connection_arguments(parser, *, required=False):
    selection = parser.add_mutually_exclusive_group(required=required)
    selection.add_argument("--serial", help="Serial number of the hand to connect")
    selection.add_argument("--address", help="SDK endpoint, HOST:PORT")
    parser.add_argument("--handedness", choices=("left", "right"), default="left")


def _plan_arguments(parser):
    parser.add_argument("gestures", nargs="+", help="Gesture names, or all")
    parser.add_argument("--model", required=True, type=Path, help="Official hand MJCF file")
    parser.add_argument("--hold", type=float, default=4.0, help="Gesture hold time in seconds")
    parser.add_argument("--speed", type=float, default=0.2, help="Peak command speed in rad/s")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("gestures", help="List gestures without connecting to a device")
    commands.add_parser("discover", help="Scan for devices without enabling motors")
    inspect = commands.add_parser("inspect", help="Read joint feedback without changing settings")
    _connection_arguments(inspect, required=True)
    check = commands.add_parser("check", help="Check gesture paths without connecting to hardware")
    _plan_arguments(check)
    check.add_argument("--handedness", choices=("left", "right"), default="left")
    check.add_argument(
        "--start-json", type=Path, help="Optional JSON array of 20 initial joint angles"
    )
    run = commands.add_parser(
        "run", help="Plan a gesture sequence; --execute enables hardware motion"
    )
    _plan_arguments(run)
    _connection_arguments(run)
    run.add_argument(
        "--execute", action="store_true", help="Execute the sequence on the selected hand"
    )
    run.add_argument("--camera", help="Camera path or numeric index; required with --execute")
    run.add_argument(
        "--output", type=Path, default=Path("recordings"), help="Recording parent directory"
    )
    run.add_argument("--kp", type=float, default=3.0)
    run.add_argument("--kd", type=float, default=0.05)
    run.add_argument("--current-limit", type=float, default=0.5, help="Current limit in amperes")
    run.add_argument(
        "--max-temperature", type=float, default=60.0, help="Session MCU cutoff in Celsius"
    )
    export = commands.add_parser("export", help="Export a recording at its measured timing")
    export.add_argument("recording", type=Path)
    export.add_argument("output", type=Path)
    export.add_argument("--fps", type=float, default=30.0)
    return parser


def _names(values):
    names = list(list_gestures()) if values == ["all"] else values
    if "all" in names:
        raise ValueError("Use all by itself, or supply individual gesture names")
    return names


def _prepare(args):
    names = _names(args.gestures)
    # The recorded pose set is for the left Beta 2 hand only.
    for name in names:
        get_pose(name, handedness=args.handedness)
    config = ControlConfig(speed_rad_s=args.speed)
    waypoints = gesture_sequence(names, hold_s=args.hold)
    model = JointModel(args.model, handedness=args.handedness)
    start_file = getattr(args, "start_json", None)
    start = json.loads(start_file.read_text()) if start_file else get_pose("open_palm")
    previous = start
    duration = 0.0
    for waypoint in waypoints:
        if waypoint.hold_s > config.max_hold_s:
            raise ValueError(f"Hold time must not exceed {config.max_hold_s:g} seconds")
        model.preflight(previous, waypoint.q)
        duration += quintic_duration(previous, waypoint.q, args.speed) + waypoint.hold_s
        previous = waypoint.q
    if duration > config.max_run_s:
        raise ValueError("Sequence exceeds the session duration; run fewer gestures together")
    plan = {
        "gestures": names,
        "handedness": args.handedness,
        "model": str(args.model.resolve()),
        "estimated_duration_s": duration,
        "initial_pose_source": str(start_file) if start_file else "open_palm",
        "waypoints": [{"name": w.name, "q": w.q.tolist(), "hold_s": w.hold_s} for w in waypoints],
    }
    return model, waypoints, plan


def _discover():
    try:
        from wuji_sdk import SdkManager
    except ImportError as exc:
        raise RuntimeError("Install wuji2-control[hardware] to use device discovery") from exc
    devices = SdkManager.instance().scan()
    return [
        {"serial": d.sn, "address": str(d.address), "device_type": str(d.device_type)}
        for d in devices
    ]


def _inspect(args):
    device = Wuji2Device(
        ConnectionConfig(
            serial=args.serial,
            address=args.address,
            handedness=args.handedness,
        )
    )
    try:
        device.connect()
        return asdict(device.wait_for_snapshot())
    finally:
        device.close()


def _run(args):
    model, waypoints, plan = _prepare(args)
    if not args.execute:
        return {"executed": False, **plan}
    if not args.camera:
        raise ValueError("--camera is required with --execute")
    connection = ConnectionConfig(
        serial=args.serial,
        address=args.address,
        handedness=args.handedness,
    )
    config = ControlConfig(
        kp=args.kp,
        kd=args.kd,
        current_limit_A=args.current_limit,
        speed_rad_s=args.speed,
        max_temperature_C=args.max_temperature,
    )
    from .recording import CameraRecorder
    from .session import ControlSession

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = args.output.resolve() / f"{stamp}_{uuid4().hex[:8]}"
    out.mkdir(parents=True)
    (out / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    report = {"completed": False, "directory": str(out), "settings": asdict(config)}
    session = None
    saved_handlers = {}
    try:
        with (out / "telemetry.jsonl").open("w", buffering=1) as telemetry:

            def log_sample(sample):
                telemetry.write(json.dumps(sample) + "\n")

            with CameraRecorder(args.camera, out) as recorder:
                recorder.snapshot("before")
                device = Wuji2Device(connection)
                session = ControlSession(
                    device,
                    model,
                    config=config,
                    guard=recorder.check,
                    on_sample=log_sample,
                )
                for sig in (signal.SIGINT, signal.SIGTERM):
                    saved_handlers[sig] = signal.getsignal(sig)
                    signal.signal(sig, lambda *_: session.stop_event.set())
                with session:
                    session.run(waypoints)
                # The session has disabled motors before the recorder is closed.
                recorder.snapshot("after")
        report.update(session.result)
        report["completed"] = True
        return report
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if session is not None:
            report.update(session.result)
        if "error" in report:
            report["completed"] = False
        for sig, previous in saved_handlers.items():
            signal.signal(sig, previous)
        (out / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"Recording: {out}", file=sys.stderr)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "gestures":
            result = {"handedness": "left", "gestures": list(list_gestures())}
        elif args.command == "discover":
            result = _discover()
        elif args.command == "inspect":
            result = _inspect(args)
        elif args.command == "check":
            _, _, result = _prepare(args)
        elif args.command == "run":
            result = _run(args)
        else:
            from .recording import export_video

            result = {"video": str(export_video(args.recording, args.output, fps=args.fps))}
        print(json.dumps(result, indent=2))
        return 0
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
