"""Completion marker of a training run: <run>/finished.json.

The trainer writes it after logging '[fine] DONE' and saving the checkpoint:

    {"format": "dsad-finished-1", "done": "<the [fine] DONE line>",
     "files": {"gaussian.pth": {"size": <int>, "sha256": "<hex>"},
               "deform_sad/iteration_<it>/deform_sad.pth": {"size": <int>, "sha256": "<hex>"}}}

The keys of `files` are paths relative to the run directory, with '/' separators.
A run has finished when finished.json exists and its files have the recorded
sizes (check_finished). The stored form (dsad/export.py) and the decode score
(decode.json, scripts/score_decode.py) record the same `files` dict as their
`source`: they belong to the current checkpoint exactly when the two are equal.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

FINISHED = "finished.json"
FORMAT = "dsad-finished-1"


def file_entry(path: Path) -> dict:
    """{"size", "sha256"} of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return {"size": Path(path).stat().st_size, "sha256": h.hexdigest()}


def files_entry(run: Path, files) -> dict:
    """{path relative to the run directory: file_entry} of checkpoint files inside it.
    Each file may be given as an absolute path, relative to the working directory, or
    relative to the run directory."""
    run = Path(run)
    root = run.resolve()
    out = {}
    for f in files:
        f = Path(f)
        if not f.is_absolute() and not f.exists() and (run / f).exists():
            f = run / f                       # given relative to the run directory
        f = f.resolve()
        out[f.relative_to(root).as_posix()] = file_entry(f)
    return out


def write_finished(run: Path, done_line: str, files) -> Path:
    """Write <run>/finished.json for the checkpoint files `files` (as files_entry takes
    them). The write is atomic."""
    run = Path(run)
    marker = {"format": FORMAT, "done": done_line.strip(), "files": files_entry(run, files)}
    tmp = run / f".{FINISHED}.tmp"
    tmp.write_text(json.dumps(marker, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, run / FINISHED)
    return run / FINISHED


def read_finished(run: Path) -> dict | None:
    """The run's finished.json, or None if it is missing, unreadable or of another format."""
    try:
        marker = json.loads((Path(run) / FINISHED).read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(marker, dict) or marker.get("format") != FORMAT or not isinstance(marker.get("files"), dict):
        return None
    return marker


def check_finished(run: Path, hashes: bool = False) -> bool:
    """True if finished.json exists and every file it names has the recorded size (with
    hashes=True, also the recorded sha256)."""
    marker = read_finished(run)
    if marker is None or not marker["files"]:
        return False
    for rel, want in marker["files"].items():
        f = Path(run) / rel
        try:
            if f.stat().st_size != want.get("size"):
                return False
        except OSError:
            return False
        if hashes and file_entry(f)["sha256"] != want.get("sha256"):
            return False
    return True
