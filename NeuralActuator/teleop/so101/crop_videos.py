#!/usr/bin/env python3
"""
Crop MP4 videos: remove left 1/4 and right 1/4 (keep center half), mute audio.
Original file is kept as xxx_original.mp4, cropped version takes the original name.

Supports single file or folder (including train/validation/test subfolders).

Usage:
    python crop_videos.py dataset/video_data/lift_and_hold/200g/003.mp4   # single file
    python crop_videos.py dataset/video_data/lift_and_hold/200g            # whole folder
"""

import subprocess
import sys
from pathlib import Path


def crop_video(video_path: Path) -> bool:
    """Crop single video: keep center 1/2, remove audio.

    003.mp4 -> 003_original.mp4 (backup) + 003.mp4 (cropped)
    """
    original_backup = video_path.with_name(f"{video_path.stem}_original{video_path.suffix}")
    temp_path = video_path.with_suffix(".tmp.mp4")

    # Skip if already cropped (backup exists)
    if original_backup.exists():
        print(f"    [SKIP] {video_path.name} — {original_backup.name} already exists")
        return True

    # crop=w:h:x:y — keep center half (remove left 1/4 and right 1/4)
    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-vf", "crop=iw/2:ih:iw/4:0",
        "-an",
        "-y",
        str(temp_path),
    ]
    result = subprocess.run(cmd, capture_output=True)

    if result.returncode == 0:
        # Rename original -> backup, temp -> original name
        video_path.rename(original_backup)
        temp_path.rename(video_path)
        return True
    else:
        if temp_path.exists():
            temp_path.unlink()
        print(f"    [ERR] ffmpeg failed: {result.stderr.decode()[-200:]}")
        return False


def process_folder(folder: Path):
    """Process all MP4 files in a folder (excluding _original and .tmp files)."""
    videos = sorted([
        v for v in folder.glob("*.mp4")
        if "_original" not in v.stem and ".tmp" not in v.stem
    ])
    if not videos:
        print(f"  No MP4 files in {folder}")
        return 0

    print(f"  {folder} — {len(videos)} files")
    ok = 0
    for i, v in enumerate(videos, 1):
        print(f"    [{i}/{len(videos)}] {v.name}", end=" ")
        if crop_video(v):
            print("OK")
            ok += 1
        else:
            print("FAILED")
    return ok


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <file_or_folder>")
        sys.exit(1)

    target = Path(sys.argv[1])

    if target.is_file() and target.suffix == ".mp4":
        print(f"Processing single file: {target.name}")
        if crop_video(target):
            print(f"  OK — original saved as {target.stem}_original.mp4")
        else:
            print("  FAILED")
        return

    if not target.is_dir():
        print(f"[ERR] Not a file or directory: {target}")
        sys.exit(1)

    # Check for train/validation/test subfolders
    subfolders = [target / sf for sf in ["train", "validation", "test"] if (target / sf).is_dir()]

    if subfolders:
        # Process each subfolder
        total = 0
        for sf in subfolders:
            total += process_folder(sf)
        print(f"\nDone. Cropped {total} files across {len(subfolders)} subfolders.")
    else:
        # Process folder directly
        total = process_folder(target)
        print(f"\nDone. Cropped {total} files.")


if __name__ == "__main__":
    main()
