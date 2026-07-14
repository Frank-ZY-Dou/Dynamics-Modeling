#!/usr/bin/env python
"""
Convert pos1-pos6 and goal_pos1-goal_pos6 from raw STS3215 ticks to radians
in existing CSV files.

Reads follower calibration from:
    ~/.cache/huggingface/lerobot/calibration/robots/so_follower/my_awesome_follower_arm.json

Formula: radians = (raw - mid) * 2π / 4095
         where mid = (range_min + range_max) / 2

Usage:
    python convert_pos_to_rad.py <csv_folder>
    python convert_pos_to_rad.py dataset/trajectory_data/lift_and_hold/200g
    python convert_pos_to_rad.py dataset/trajectory_data/lift_and_hold/200g/001.csv   # single file
"""

import csv
import json
import os
import sys
from math import pi
from pathlib import Path

NUM_MOTORS = 6
STS3215_RESOLUTION = 4095

# Default calibration path
DEFAULT_CALIB = Path.home() / ".cache/huggingface/lerobot/calibration/robots/so_follower/my_awesome_follower_arm.json"

POS_COLS = [f"pos{i}" for i in range(1, NUM_MOTORS + 1)]
GOAL_COLS = [f"goal_pos{i}" for i in range(1, NUM_MOTORS + 1)]


def load_mids(calib_path):
    """Load calibration JSON and compute mid-point for each motor."""
    with open(calib_path) as f:
        calib = json.load(f)

    # Return mids in motor order (shoulder_pan=1, ..., gripper=6)
    motor_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    mids = []
    for name in motor_names:
        rmin = calib[name]["range_min"]
        rmax = calib[name]["range_max"]
        mids.append((rmin + rmax) / 2.0)
    return mids


def raw_to_rad(raw_val, mid):
    return (raw_val - mid) * 2.0 * pi / STS3215_RESOLUTION


def process_file(filepath, mids):
    with open(filepath, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    # Find column indices
    pos_indices = []
    goal_indices = []
    for i, col in enumerate(POS_COLS):
        if col in header:
            pos_indices.append((header.index(col), i))
    for i, col in enumerate(GOAL_COLS):
        if col in header:
            goal_indices.append((header.index(col), i))

    if not pos_indices and not goal_indices:
        print(f"[SKIP] {filepath} — no pos/goal_pos columns found")
        return

    # Check if already converted (first data row: if values are small floats, likely already radians)
    if rows:
        first_pos_val = float(rows[0][pos_indices[0][0]]) if pos_indices else 0
        if -10.0 < first_pos_val < 10.0:
            print(f"[SKIP] {filepath} — values look like radians already (first pos = {first_pos_val:.4f})")
            return

    # Convert
    for row in rows:
        for col_idx, motor_idx in pos_indices:
            row[col_idx] = f"{raw_to_rad(float(row[col_idx]), mids[motor_idx]):.6f}"
        for col_idx, motor_idx in goal_indices:
            row[col_idx] = f"{raw_to_rad(float(row[col_idx]), mids[motor_idx]):.6f}"

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"[DONE] {filepath} — converted {len(pos_indices)} pos + {len(goal_indices)} goal_pos cols to radians ({len(rows)} rows)")


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <csv_folder_or_file> [--calib path/to/calib.json]")
        sys.exit(1)

    # Parse optional --calib argument
    calib_path = DEFAULT_CALIB
    args = sys.argv[1:]
    if "--calib" in args:
        idx = args.index("--calib")
        calib_path = Path(args[idx + 1])
        args = args[:idx] + args[idx + 2:]

    target = args[0]

    if not calib_path.is_file():
        print(f"[ERR] Calibration file not found: {calib_path}")
        sys.exit(1)

    mids = load_mids(calib_path)
    print(f"Calibration: {calib_path}")
    print(f"Motor mids: {[f'{m:.1f}' for m in mids]}\n")

    if os.path.isfile(target) and target.endswith(".csv"):
        process_file(target, mids)
    elif os.path.isdir(target):
        csv_files = sorted(f for f in os.listdir(target) if f.endswith(".csv"))
        if not csv_files:
            print(f"[ERR] No CSV files in {target}")
            sys.exit(1)
        print(f"Processing {len(csv_files)} files in {target}/\n")
        for f in csv_files:
            process_file(os.path.join(target, f), mids)
    else:
        print(f"[ERR] {target} — not a CSV file or directory")
        sys.exit(1)


if __name__ == "__main__":
    main()
