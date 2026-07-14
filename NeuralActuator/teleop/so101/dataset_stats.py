#!/usr/bin/env python3
"""Compute frame counts and total duration for CSV dataset files.

Supports two directory layouts:
  1. Direct split:   root_dir/train/*.csv, root_dir/validation/*.csv, root_dir/test/*.csv
  2. Nested weights: root_dir/200g/train/*.csv, root_dir/300g/train/*.csv, ...

Usage:
    python dataset_stats.py dataset/trajectory_data/lift_and_hold/200g
    python dataset_stats.py dataset/trajectory_data/lift_and_hold          # scans all weight subfolders
"""

import argparse
from pathlib import Path

import pandas as pd


SPLIT_NAMES = ["train", "test", "validation"]


def stats_for_split_dir(root_path: Path) -> dict:
    """Compute stats for a single directory containing train/test/validation splits.

    Returns:
        {"by_split": {split: {files, frames, duration}}, "total_frames", "total_duration"}
    """
    total_frames = 0
    total_duration = 0.0
    stats_by_split = {}

    for split in SPLIT_NAMES:
        split_path = root_path / split
        if not split_path.exists():
            continue

        split_frames = 0
        split_duration = 0.0
        csv_count = 0

        for csv_file in sorted(split_path.glob("*.csv")):
            try:
                df = pd.read_csv(csv_file)
                num_rows = len(df)
                split_frames += num_rows

                if "timestamp" in df.columns and num_rows > 0:
                    split_duration += df["timestamp"].iloc[-1]
                else:
                    print(f"  [WARN] {csv_file} has no 'timestamp' column or is empty")

                csv_count += 1
            except Exception as e:
                print(f"  [ERR] {csv_file}: {e}")

        stats_by_split[split] = {
            "files": csv_count,
            "frames": split_frames,
            "duration": split_duration,
        }
        total_frames += split_frames
        total_duration += split_duration

    return {
        "by_split": stats_by_split,
        "total_frames": total_frames,
        "total_duration": total_duration,
    }


def has_splits(path: Path) -> bool:
    """Check if a directory directly contains train/test/validation subfolders."""
    return any((path / s).is_dir() for s in SPLIT_NAMES)


def print_stats(name: str, stats: dict):
    """Pretty-print stats for one dataset group."""
    print(f"\n  {name}:")
    for split, s in stats["by_split"].items():
        print(f"    {split:12s}  {s['files']} files, {s['frames']} frames, {s['duration']:.1f}s")
    print(f"    {'total':12s}  {stats['total_frames']} frames, "
          f"{stats['total_duration']:.1f}s ({stats['total_duration']/60:.1f} min)")


def main():
    parser = argparse.ArgumentParser(description="Compute frame counts and duration for CSV dataset")
    parser.add_argument("root_dir", type=str,
                        help="Dataset root (with train/test/validation, or with weight subfolders)")
    args = parser.parse_args()

    root = Path(args.root_dir)
    if not root.exists():
        print(f"[ERR] Directory not found: {root}")
        return

    print("=" * 60)
    print(f"Dataset Statistics: {root}")
    print("=" * 60)

    grand_frames = 0
    grand_duration = 0.0

    if has_splits(root):
        # Layout 1: root_dir directly has train/test/validation
        stats = stats_for_split_dir(root)
        print_stats(root.name, stats)
        grand_frames = stats["total_frames"]
        grand_duration = stats["total_duration"]
    else:
        # Layout 2: root_dir has subfolders (200g, 300g, ...) each with splits
        subdirs = sorted([d for d in root.iterdir() if d.is_dir() and has_splits(d)])

        if not subdirs:
            # No splits found anywhere, try scanning CSV files directly
            csv_files = sorted(root.rglob("*.csv"))
            if csv_files:
                print(f"\n  No train/test/validation splits found.")
                print(f"  Found {len(csv_files)} CSV files total (unsplit).")
                for f in csv_files:
                    df = pd.read_csv(f)
                    dur = df["timestamp"].iloc[-1] if "timestamp" in df.columns and len(df) > 0 else 0
                    print(f"    {f.relative_to(root)}  {len(df)} frames, {dur:.1f}s")
                    grand_frames += len(df)
                    grand_duration += dur
            else:
                print("  No CSV files found.")
                return
        else:
            for sub in subdirs:
                stats = stats_for_split_dir(sub)
                if stats["total_frames"] > 0:
                    print_stats(sub.name, stats)
                    grand_frames += stats["total_frames"]
                    grand_duration += stats["total_duration"]

    print("\n" + "-" * 60)
    print(f"Grand Total: {grand_frames} frames, {grand_duration:.1f}s ({grand_duration/60:.1f} min)")
    print("=" * 60)


if __name__ == "__main__":
    main()
