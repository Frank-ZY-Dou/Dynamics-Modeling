#!/usr/bin/env python3
"""
Split dataset into train/validation/test, synced between trajectory_data and video_data.

Randomly picks 1 file for test, 1 for validation, rest for train.
Train files are renumbered in original order (001, 002, ...).
Automatically finds the matching video_data folder.

Directory structure before:
    dataset/trajectory_data/lift_and_hold/200g/  001.csv ~ 010.csv
    dataset/video_data/lift_and_hold/200g/       001.mp4 ~ 010.mp4

Directory structure after:
    dataset/trajectory_data/lift_and_hold/200g/
        train/       001.csv 002.csv ... 008.csv  (renumbered in original order)
        validation/  001.csv
        test/        001.csv
    dataset/video_data/lift_and_hold/200g/
        train/       001.mp4 002.mp4 ... 008.mp4
        validation/  001.mp4
        test/        001.mp4

Usage:
    python split_dataset.py dataset/trajectory_data/lift_and_hold/200g
    python split_dataset.py dataset/trajectory_data/lift_and_hold/200g --seed 42
    python split_dataset.py dataset/trajectory_data/lift_and_hold/200g --copy
"""

import argparse
import random
import shutil
from pathlib import Path


def find_video_dir(csv_dir: Path) -> Path | None:
    """Derive video_data path from trajectory_data path."""
    csv_str = str(csv_dir)
    if "trajectory_data" in csv_str:
        video_str = csv_str.replace("trajectory_data", "video_data")
        video_path = Path(video_str)
        if video_path.exists():
            return video_path
    return None


def split_dataset(csv_dir: str, seed: int = None, copy: bool = False):
    csv_path = Path(csv_dir)
    if not csv_path.exists():
        raise ValueError(f"Directory not found: {csv_dir}")

    video_path = find_video_dir(csv_path)

    # Collect CSV files
    csv_files = sorted([f for f in csv_path.iterdir() if f.is_file() and f.suffix == ".csv"])
    if len(csv_files) < 3:
        raise ValueError(f"Need at least 3 CSV files, found {len(csv_files)}")

    # Collect video files by stem
    video_files = {}
    if video_path:
        video_files = {f.stem: f for f in video_path.iterdir() if f.is_file()}
        print(f"CSV dir:   {csv_path}  ({len(csv_files)} files)")
        print(f"Video dir: {video_path}  ({len(video_files)} files)")
    else:
        print(f"CSV dir:   {csv_path}  ({len(csv_files)} files)")
        print(f"Video dir: (not found, skipping)")

    # Get stems sorted by original number
    stems = [f.stem for f in csv_files]  # already sorted: ['001','002',...]

    # Shuffle for random split
    if seed is not None:
        random.seed(seed)
        print(f"Seed: {seed}")

    shuffled = stems.copy()
    random.shuffle(shuffled)

    test_stem = shuffled[0]
    val_stem = shuffled[1]
    # Train stems sorted in original order
    train_stems = sorted(shuffled[2:], key=lambda s: stems.index(s))

    print(f"\nSplit:")
    print(f"  Test (1):       {test_stem}")
    print(f"  Validation (1): {val_stem}")
    print(f"  Train ({len(train_stems)}):      {train_stems}")

    # Helper to process one directory
    def process_dir(base_path, ext):
        file_map = {f.stem: f for f in base_path.iterdir() if f.is_file() and f.suffix == ext}

        train_dir = base_path / "train"
        val_dir = base_path / "validation"
        test_dir = base_path / "test"
        for d in [train_dir, val_dir, test_dir]:
            d.mkdir(exist_ok=True)

        op = shutil.copy2 if copy else shutil.move
        op_name = "Copy" if copy else "Move"

        # Test
        src = file_map[test_stem]
        dst = test_dir / f"001{ext}"
        op(src, dst)
        print(f"  {src.name} -> test/001{ext}")

        # Validation
        src = file_map[val_stem]
        dst = val_dir / f"001{ext}"
        op(src, dst)
        print(f"  {src.name} -> validation/001{ext}")

        # Train (renumbered in original order)
        for i, stem in enumerate(train_stems, start=1):
            src = file_map[stem]
            dst = train_dir / f"{i:03d}{ext}"
            op(src, dst)
            print(f"  {src.name} -> train/{i:03d}{ext}")

    # Process CSV
    print(f"\n{'Copying' if copy else 'Moving'} CSV files...")
    process_dir(csv_path, ".csv")

    # Process video
    if video_path and video_files:
        # Check all stems exist in video
        missing = [s for s in stems if s not in video_files]
        if missing:
            print(f"\n[WARN] Video files missing for stems: {missing}, skipping video split")
        else:
            # Detect video extension from first file
            vid_ext = video_files[stems[0]].suffix  # e.g. ".mp4"
            print(f"\n{'Copying' if copy else 'Moving'} video files...")
            process_dir(video_path, vid_ext)

    print("\nDone!")


def main():
    parser = argparse.ArgumentParser(description="Split dataset (CSV + video) into train/validation/test")
    parser.add_argument("csv_dir", help="Folder with CSV files (e.g. dataset/trajectory_data/lift_and_hold/200g)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--copy", action="store_true", help="Copy instead of move")
    args = parser.parse_args()
    split_dataset(args.csv_dir, seed=args.seed, copy=args.copy)


if __name__ == "__main__":
    main()
