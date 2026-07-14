#!/usr/bin/env python
"""
Set force_z column to a specified value for all CSVs in a folder.
Designed for lift_and_hold data where the held weight is known.

Usage:
    python lift_and_hold_set_force_z.py <csv_folder> <value>
    python lift_and_hold_set_force_z.py dataset/trajectory_data/lift_and_hold/200g -1.962
    python lift_and_hold_set_force_z.py dataset/trajectory_data/lift_and_hold/500g -4.905
"""

import csv
import os
import sys


def process_file(filepath, value):
    with open(filepath, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    if "force_z" not in header:
        print(f"[SKIP] {filepath} — no force_z column")
        return

    idx = header.index("force_z")
    for row in rows:
        row[idx] = str(value)

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"[DONE] {filepath} — force_z = {value} ({len(rows)} rows)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <csv_folder> <force_z_value>")
        sys.exit(1)

    folder = sys.argv[1]
    value = sys.argv[2]

    if not os.path.isdir(folder):
        print(f"[ERR] {folder} — not a directory")
        sys.exit(1)

    csv_files = sorted(f for f in os.listdir(folder) if f.endswith(".csv"))
    if not csv_files:
        print(f"[ERR] No CSV files found in {folder}")
        sys.exit(1)

    print(f"Setting force_z = {value} for {len(csv_files)} files in {folder}/\n")
    for f in csv_files:
        process_file(os.path.join(folder, f), value)
