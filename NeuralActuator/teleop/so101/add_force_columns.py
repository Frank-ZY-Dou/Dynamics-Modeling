#!/usr/bin/env python
"""
Add force_x, force_y, force_z columns to a CSV file if missing.
Values default to 0.

Usage:
    python add_force_columns.py path/to/data.csv
    python add_force_columns.py path/to/*.csv          # batch
"""

import csv
import sys
import os

FORCE_COLS = ["force_x", "force_y", "force_z"]


def process_file(filepath):
    with open(filepath, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    missing = [c for c in FORCE_COLS if c not in header]
    if not missing:
        print(f"[SKIP] {filepath} — already has force_x, force_y, force_z")
        return

    print(f"[ADD]  {filepath} — adding {missing}")
    for col in FORCE_COLS:
        if col not in header:
            header.append(col)
            for row in rows:
                row.append("-999")

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <csv_file> [csv_file2 ...]")
        sys.exit(1)

    for path in sys.argv[1:]:
        if not os.path.isfile(path):
            print(f"[ERR]  {path} — not found")
            continue
        process_file(path)
