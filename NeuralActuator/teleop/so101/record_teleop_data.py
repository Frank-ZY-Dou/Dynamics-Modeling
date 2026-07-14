#!/usr/bin/env python
"""
LeRobot SO101 Teleoperation Data Recorder

Records all STS3215 sensor data during leader-follower teleoperation,
saving CSV + video in a format matching the OpenManipulator data collection
(see teleop/omx/).

Two-phase workflow (same as OpenManipulator twin_motion):
    Phase 1: Teleop only — follower tracks leader, no data saved
    Phase 2: Press ENTER → CSV + video recording starts (synchronized)

Directory structure (created next to this script, i.e. lerobot_so101/dataset/):
    dataset/
    ├── trajectory_data/
    │   ├── lift_and_hold/{200g,300g,...}/  001.csv, 002.csv ...
    │   └── pick_and_place/{200g,300g,...}/ 001.csv, 002.csv ...
    └── video_data/
        └── (same mirror structure)          001.mp4, 002.mp4 ...

Usage (run from this directory, lerobot_so101/):
    conda activate lerobot
    python record_teleop_data.py
    python record_teleop_data.py --no-video
    python record_teleop_data.py --duration 60
    python record_teleop_data.py --zoom 2.0           # 2x digital zoom
    python record_teleop_data.py --res 1080p           # resolution preset
"""

import argparse
import csv
import os
import select
import signal
import sys
import threading
import time

from math import pi

import cv2
import numpy as np

# The custom SO101 leader/follower classes live in the lerobot fork. lerobot is
# normally pip-installed (editable) in the `lerobot` conda env, so `import lerobot`
# works directly. As a fallback (e.g. non-editable install), also add the lerobot
# repo's src to the path. Override with the LEROBOT_SRC env var if the repo moved.
_LEROBOT_SRC = os.environ.get("LEROBOT_SRC", "")  # set to your lerobot src/ if not pip-installed
for _p in (os.path.join(os.path.dirname(__file__), "src"), _LEROBOT_SRC):
    if os.path.isdir(_p):
        sys.path.insert(0, _p)

from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.robots.so_follower.so_follower import SOFollower
from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig
from lerobot.teleoperators.so_leader.so_leader import SOLeader

# ── Constants ──
MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
NUM_MOTORS = len(MOTOR_NAMES)
TARGET_DT = 0.016  # 16ms → ~62.5 Hz

SENSOR_REGISTERS = [
    "Present_Position",
    "Present_Velocity",
    "Present_Load",
    "Present_Voltage",
    "Present_Temperature",
    "Present_Current",
]

# Base dataset directory
DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
BASE_DIR = os.path.join(DATASET_DIR, "trajectory_data")
VIDEO_DIR = os.path.join(DATASET_DIR, "video_data")

# Category definitions
CATEGORIES = {
    # Lift and hold
    "1":  "lift_and_hold/200g",
    "2":  "lift_and_hold/300g",
    "3":  "lift_and_hold/400g",
    "4":  "lift_and_hold/500g",
    "5":  "lift_and_hold/empty",
    # Pick and place
    "6":  "pick_and_place/200g",
    "7":  "pick_and_place/300g",
    "8":  "pick_and_place/400g",
    "9":  "pick_and_place/500g",
    "10": "pick_and_place/empty",
}

# Resolution presets (matching OpenManipulator dxl_arm_class.py)
RESOLUTION_PRESETS = {
    "4k":    (3840, 2160),
    "qhd":   (2560, 1440),
    "2k":    (2560, 1440),
    "1080p": (1920, 1080),
    "720p":  (1280, 720),
}


# ═══════════════════════════════════════════════════════════════
#  File path helpers (matching OpenManipulator get_trajectory_filepath)
# ═══════════════════════════════════════════════════════════════

def get_trajectory_filepath(with_video=True):
    """Interactive category selection + auto-numbered filepath generation."""
    print("\n" + "=" * 60)
    print("  SELECT TRAJECTORY CATEGORY")
    print("=" * 60)
    print("\n[Lift and hold]")
    print("  1.  200g")
    print("  2.  300g")
    print("  3.  400g")
    print("  4.  500g")
    print("  5.  empty")
    print("\n[Pick and place]")
    print("  6.  200g")
    print("  7.  300g")
    print("  8.  400g")
    print("  9.  500g")
    print("  10. empty")
    print("\n  0. Custom path")
    print("=" * 60)

    choice = input("Enter category number: ").strip()

    if choice == "0":
        print(f"\nBase directory: {BASE_DIR}/")
        custom_input = input("Enter subfolder (relative to above), or absolute path: ").strip()
        if not custom_input:
            csv_dir = os.path.join(BASE_DIR, "custom")
            video_subdir = "custom"
        elif os.path.isabs(custom_input):
            csv_dir = custom_input
            video_subdir = custom_input
        else:
            csv_dir = os.path.join(BASE_DIR, custom_input)
            video_subdir = custom_input

        os.makedirs(csv_dir, exist_ok=True)
        filename = _next_numbered_filename(csv_dir)
        csv_filepath = os.path.join(csv_dir, filename)
        print(f"\n>>> CSV will save to: {csv_filepath}")

        if with_video:
            if os.path.isabs(custom_input) if custom_input else False:
                vid_dir = csv_dir.replace("trajectory_data", "video_data")
            else:
                vid_dir = os.path.join(VIDEO_DIR, video_subdir)
            os.makedirs(vid_dir, exist_ok=True)
            video_filepath = os.path.join(vid_dir, filename.replace(".csv", ".mp4"))
            print(f">>> Video will save to: {video_filepath}")
            return csv_filepath, video_filepath
        return (csv_filepath,)

    if choice not in CATEGORIES:
        print(f"Invalid choice '{choice}', defaulting to 1 (lift_and_hold/200g)")
        choice = "1"

    subdir = CATEGORIES[choice]
    csv_dir = os.path.join(BASE_DIR, subdir)
    os.makedirs(csv_dir, exist_ok=True)

    existing_files = [f for f in os.listdir(csv_dir) if f.endswith(".csv")]
    print(f"Found {len(existing_files)} existing files in {subdir}/")

    filename = _next_numbered_filename(csv_dir)
    csv_filepath = os.path.join(csv_dir, filename)
    print(f"\n>>> CSV will save to: {csv_filepath}")

    if with_video:
        vid_dir = os.path.join(VIDEO_DIR, subdir)
        os.makedirs(vid_dir, exist_ok=True)
        video_filepath = os.path.join(vid_dir, filename.replace(".csv", ".mp4"))
        print(f">>> Video will save to: {video_filepath}")
        return csv_filepath, video_filepath

    return (csv_filepath,)


def _next_numbered_filename(directory):
    existing_nums = []
    for f in os.listdir(directory):
        if f.endswith(".csv"):
            try:
                existing_nums.append(int(f[:-4]))
            except ValueError:
                pass
    next_num = max(existing_nums, default=0) + 1

    print(f"Next auto number: {next_num:03d}")
    custom_name = input(f"Enter filename (or press Enter for '{next_num:03d}'): ").strip()

    if custom_name:
        if not custom_name.endswith(".csv"):
            custom_name += ".csv"
        return custom_name
    return f"{next_num:03d}.csv"


# ═══════════════════════════════════════════════════════════════
#  CSV helpers
# ═══════════════════════════════════════════════════════════════

def build_csv_header():
    header = ["timestamp"]
    header += [f"pos{i}" for i in range(1, NUM_MOTORS + 1)]
    header += [f"goal_pos{i}" for i in range(1, NUM_MOTORS + 1)]
    header += [f"current{i}" for i in range(1, NUM_MOTORS + 1)]
    header += [f"vel{i}" for i in range(1, NUM_MOTORS + 1)]
    header += [f"load{i}" for i in range(1, NUM_MOTORS + 1)]
    header += [f"temp{i}" for i in range(1, NUM_MOTORS + 1)]
    header += [f"volts{i}" for i in range(1, NUM_MOTORS + 1)]
    header += ["force_x", "force_y", "force_z"]
    return header


def ordered_values(d):
    return [d.get(m, 0) for m in MOTOR_NAMES]


DEG_TO_RAD = pi / 180.0


def read_positions_rad(bus):
    """Read Present_Position and Goal_Position in radians via normalize=True (degrees) then convert."""
    result = {}
    for reg in ["Present_Position", "Goal_Position"]:
        try:
            deg_dict = bus.sync_read(reg)  # normalize=True → degrees
            result[reg] = {m: v * DEG_TO_RAD for m, v in deg_dict.items()}
        except Exception as e:
            print(f"[WARN] Failed to read {reg}: {e}")
            result[reg] = {m: 0.0 for m in MOTOR_NAMES}
    return result


def read_follower_sensors(bus):
    sensors = {}
    for reg in SENSOR_REGISTERS + ["Goal_Position"]:
        try:
            sensors[reg] = bus.sync_read(reg, normalize=False)
        except Exception as e:
            print(f"[WARN] Failed to read {reg}: {e}")
            sensors[reg] = {m: 0 for m in MOTOR_NAMES}
    return sensors


def save_csv(csv_path, data_rows):
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(build_csv_header())
        writer.writerows(data_rows)


# ═══════════════════════════════════════════════════════════════
#  Digital zoom helper
# ═══════════════════════════════════════════════════════════════

def apply_zoom(frame, zoom_factor):
    """Crop center of frame and resize back to original dimensions (digital zoom)."""
    if zoom_factor <= 1.0:
        return frame
    h, w = frame.shape[:2]
    # Crop region
    crop_h = int(h / zoom_factor)
    crop_w = int(w / zoom_factor)
    y0 = (h - crop_h) // 2
    x0 = (w - crop_w) // 2
    cropped = frame[y0:y0 + crop_h, x0:x0 + crop_w]
    # Resize back to original dimensions
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


# ═══════════════════════════════════════════════════════════════
#  Video recorder (background thread, synced with CSV via Barrier)
# ═══════════════════════════════════════════════════════════════

class VideoRecorder:
    """Records video in a background thread, synchronized with CSV via Barrier."""

    def __init__(self, camera, video_path, fps=30, zoom=1.0):
        self.camera = camera
        self.video_path = video_path
        self.fps = fps
        self.zoom = zoom
        self.writer = None
        self.thread = None
        self.stop_event = threading.Event()
        self.recording_event = threading.Event()
        self.start_barrier = threading.Barrier(2, timeout=10)
        self.start_time = None
        self.frame_count = 0

    def start(self):
        self.thread = threading.Thread(target=self._record_loop, daemon=True)
        self.thread.start()

    def trigger_record(self):
        self.recording_event.set()

    def wait_for_sync(self):
        self.start_barrier.wait()
        self.start_time = time.time()
        return self.start_time

    def _read_frame(self):
        """Read a frame from camera, apply zoom."""
        frame = self.camera.read_latest(max_age_ms=500)
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if self.zoom > 1.0:
            bgr = apply_zoom(bgr, self.zoom)
        return bgr

    def _record_loop(self):
        dt = 1.0 / self.fps
        self.recording_event.wait()

        # Pre-warm: open VideoWriter before barrier
        bgr = None
        for _ in range(50):
            try:
                bgr = self._read_frame()
                break
            except Exception:
                time.sleep(0.02)
        if bgr is None:
            print("[VIDEO] ERROR: Could not get camera frame, aborting video.")
            try:
                self.start_barrier.wait()
            except threading.BrokenBarrierError:
                pass
            return

        h, w = bgr.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(self.video_path, fourcc, self.fps, (w, h))
        if not self.writer.isOpened():
            print("[VIDEO] WARNING: Failed to open VideoWriter")
            try:
                self.start_barrier.wait()
            except threading.BrokenBarrierError:
                pass
            return

        # Barrier: simultaneous t=0 with main thread
        try:
            self.start_barrier.wait()
        except threading.BrokenBarrierError:
            return

        # Write pre-read frame as frame #0
        self.writer.write(bgr)
        self.frame_count += 1

        while not self.stop_event.is_set():
            loop_start = time.perf_counter()
            try:
                bgr = self._read_frame()
            except Exception:
                continue

            self.writer.write(bgr)
            self.frame_count += 1

            elapsed = time.perf_counter() - loop_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=3)
        if self.writer is not None:
            self.writer.release()
        print(f"[VIDEO] Saved {self.frame_count} frames to {self.video_path}")


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="SO101 Teleop Data Recorder")
    parser.add_argument("--follower-port", default="/dev/ttyACM1")
    parser.add_argument("--leader-port", default="/dev/ttyACM0")
    parser.add_argument("--follower-id", default="my_awesome_follower_arm")
    parser.add_argument("--leader-id", default="my_awesome_leader_arm")
    parser.add_argument("--duration", type=float, default=None,
                        help="Recording duration in seconds (default: Ctrl+C to stop)")
    parser.add_argument("--no-video", action="store_true", help="Disable video recording")
    parser.add_argument("--camera-index", default="/dev/video0",
                        help="Camera device path or index")
    parser.add_argument("--res", default="qhd",
                        help="Video resolution: qhd (default, 2560x1440), 4k, 1080p, 720p, or WxH")
    parser.add_argument("--video-fps", type=int, default=30, help="Video FPS")
    parser.add_argument("--fourcc", default="MJPG",
                        help="Camera FOURCC codec (default: MJPG for high-res 30fps)")
    parser.add_argument("--zoom", type=float, default=1.0,
                        help="Digital zoom factor (e.g. 1.5, 2.0). Crops center and scales up.")
    args = parser.parse_args()

    try:
        cam_index = int(args.camera_index)
    except ValueError:
        cam_index = args.camera_index

    # Resolve resolution
    if args.res.lower() in RESOLUTION_PRESETS:
        vid_w, vid_h = RESOLUTION_PRESETS[args.res.lower()]
    else:
        vid_w, vid_h = [int(x) for x in args.res.split("x")]

    # ── Interactive filepath selection ──
    record_video = not args.no_video
    paths = get_trajectory_filepath(with_video=record_video)
    if record_video:
        csv_path, video_path = paths
    else:
        csv_path = paths[0]
        video_path = None

    print("\n" + "=" * 60)
    print("  SO101 Teleop Data Recorder")
    print("=" * 60)
    print(f"  Follower:    {args.follower_port} (ID: {args.follower_id})")
    print(f"  Leader:      {args.leader_port} (ID: {args.leader_id})")
    print(f"  CSV:         {csv_path}")
    if record_video:
        print(f"  Video:       {video_path}")
        print(f"  Camera:      {args.camera_index} ({vid_w}x{vid_h} @ {args.video_fps}fps)")
        if args.zoom > 1.0:
            print(f"  Zoom:        {args.zoom}x (digital)")
    print(f"  Sensor rate: ~62.5 Hz ({TARGET_DT * 1000:.0f}ms)")
    if args.duration:
        print(f"  Duration:    {args.duration}s")
    else:
        print("  Duration:    Manual (Ctrl+C to stop)")
    print("=" * 60)

    # ── 1. Connect camera ──
    camera = None
    video_recorder = None
    if record_video:
        print("\n[1/3] Connecting camera...")
        cam_config = OpenCVCameraConfig(
            index_or_path=cam_index,
            fps=args.video_fps,
            width=vid_w,
            height=vid_h,
            fourcc=args.fourcc,
        )
        camera = OpenCVCamera(cam_config)
        camera.connect()
        print(f"  Camera connected: {vid_w}x{vid_h} @ {args.video_fps}fps")
    else:
        print("\n[1/3] Camera: skipped (--no-video)")

    # ── 2. Connect follower ──
    print("[2/3] Connecting follower arm...")
    follower_config = SOFollowerRobotConfig(
        port=args.follower_port,
        id=args.follower_id,
    )
    follower = SOFollower(follower_config)
    follower.connect()
    print("  Follower connected (position unit: radians).")

    # ── 3. Connect leader ──
    print("[3/3] Connecting leader arm...")
    leader_config = SOLeaderTeleopConfig(
        port=args.leader_port,
        id=args.leader_id,
    )
    leader = SOLeader(leader_config)
    leader.connect()
    print("  Leader connected.")

    # ── Signal handler ──
    stop_flag = False

    def signal_handler(sig, frame):
        nonlocal stop_flag
        print("\n>>> Ctrl+C received, stopping...")
        stop_flag = True

    signal.signal(signal.SIGINT, signal_handler)

    # ══════════════════════════════════════════════════════════
    #  Phase 1: Teleop only (no data recording)
    #  Follower tracks leader. User can practice / position arm.
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  TELEOP ACTIVE — Follower is tracking leader")
    print("  Move leader arm freely. No data is being saved.")
    print("  Press ENTER to start recording CSV/video...")
    print("=" * 60)

    try:
        while True:
            iter_start = time.perf_counter()

            # Teleop only — no sensor reads, no CSV
            action = leader.get_action()
            follower.send_action(action)

            # Check for Enter key (non-blocking)
            if select.select([sys.stdin], [], [], 0)[0]:
                sys.stdin.readline()
                break

            iter_elapsed = time.perf_counter() - iter_start
            sleep_time = TARGET_DT - iter_elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    except KeyboardInterrupt:
        print("\n>>> Ctrl+C during teleop phase, exiting.")
        follower.disconnect()
        leader.disconnect()
        if camera is not None:
            camera.disconnect()
        return

    # ══════════════════════════════════════════════════════════
    #  Phase 2: Teleop + Recording (CSV + video synchronized)
    # ══════════════════════════════════════════════════════════
    if record_video and camera is not None:
        video_recorder = VideoRecorder(camera, video_path, fps=args.video_fps, zoom=args.zoom)
        video_recorder.start()
        video_recorder.trigger_record()
        csv_start_time = video_recorder.wait_for_sync()
    else:
        csv_start_time = time.time()
    print(">>> Recording started!")

    data_rows = []
    sample_count = 0

    print("  Press ENTER or Ctrl+C to stop recording.\n")

    try:
        while not stop_flag:
            iter_start = time.perf_counter()

            elapsed = time.time() - csv_start_time
            if args.duration and elapsed >= args.duration:
                print(f"\n>>> Duration {args.duration}s reached.")
                break

            # Check for Enter key (non-blocking)
            if select.select([sys.stdin], [], [], 0)[0]:
                sys.stdin.readline()
                print("\n>>> ENTER pressed, stopping recording...")
                break

            # Teleop
            action = leader.get_action()
            follower.send_action(action)

            # Read sensors
            sensors = read_follower_sensors(follower.bus)   # raw: current, vel, load, temp, volts
            positions = read_positions_rad(follower.bus)     # radians: pos, goal_pos

            # Build row
            timestamp = time.time() - csv_start_time
            row = [timestamp]
            row += ordered_values(positions["Present_Position"])
            row += ordered_values(positions["Goal_Position"])
            row += ordered_values(sensors["Present_Current"])
            row += ordered_values(sensors["Present_Velocity"])
            row += ordered_values(sensors["Present_Load"])
            row += ordered_values(sensors["Present_Temperature"])
            row += ordered_values(sensors["Present_Voltage"])
            row += [-999, -999, -999]  # force_x, force_y, force_z (no sensor)

            data_rows.append(row)
            sample_count += 1

            if sample_count % 100 == 0:
                actual_hz = sample_count / (time.time() - csv_start_time)
                print(f"  [{elapsed:.1f}s] {sample_count} samples ({actual_hz:.1f} Hz)")

            # Rate control
            iter_elapsed = time.perf_counter() - iter_start
            sleep_time = TARGET_DT - iter_elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        total_time = time.time() - csv_start_time
        actual_hz = sample_count / total_time if total_time > 0 else 0

        # Save CSV
        print(f"\n>>> Saving CSV ({sample_count} samples, {actual_hz:.1f} Hz avg)...")
        save_csv(csv_path, data_rows)
        print(f">>> CSV saved: {csv_path}")

        # Stop video
        if video_recorder is not None:
            video_recorder.stop()
            print(f">>> Video saved: {video_path}")

        # Disconnect
        print("\nDisconnecting...")
        try:
            follower.disconnect()
        except Exception:
            pass
        try:
            leader.disconnect()
        except Exception:
            pass
        if camera is not None:
            try:
                camera.disconnect()
            except Exception:
                pass

        print(f"\nDone. {sample_count} samples at {actual_hz:.1f} Hz.")
        print(f"  CSV:   {csv_path}")
        if video_path:
            print(f"  Video: {video_path}")


if __name__ == "__main__":
    main()
