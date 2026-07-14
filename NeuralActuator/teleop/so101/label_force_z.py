#!/usr/bin/env python
"""
Label force_z for a time range in a CSV file.

Interactive: prompts for start/end timestamps and force value.
Rows where start <= timestamp <= end get force_z set to the given value.

Usage:
    python label_force_z.py dataset/trajectory_data/pick_and_place/200g/001.csv
"""

import csv
import sys
import os


def main():
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <csv_file>")
        sys.exit(1)

    filepath = sys.argv[1]
    if not os.path.isfile(filepath):
        print(f"[ERR] {filepath} — not found")
        sys.exit(1)

    # Read CSV
    with open(filepath, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    if "timestamp" not in header or "force_z" not in header:
        print(f"[ERR] CSV must have 'timestamp' and 'force_z' columns")
        sys.exit(1)

    ts_idx = header.index("timestamp")
    fz_idx = header.index("force_z")

    # Show time range
    timestamps = [float(row[ts_idx]) for row in rows]
    print(f"File: {filepath}")
    print(f"Rows: {len(rows)}")
    print(f"Time range: {timestamps[0]:.4f} ~ {timestamps[-1]:.4f} s")

    # User input
    t_start = float(input("\nStart timestamp: ").strip())
    t_end = float(input("End timestamp:   ").strip())
    force_val = input("Force_z value:   ").strip()

    if t_start > t_end:
        print("[ERR] Start must be <= End")
        sys.exit(1)

    # Apply
    count = 0
    for row in rows:
        t = float(row[ts_idx])
        if t_start <= t <= t_end:
            row[fz_idx] = force_val
            count += 1

    # Save
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"\n[DONE] Set force_z = {force_val} for {count} rows ({t_start:.4f} ~ {t_end:.4f} s)")


if __name__ == "__main__":
    main()
