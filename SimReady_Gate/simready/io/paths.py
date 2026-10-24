"""Asset paths inside and outside the RoboLab checkout and the SimReady asset cache.

Layouts, certificates and export manifests write RoboLab asset paths with a `<ROBOLAB_DIR>/`
prefix and SimReady packages with a `<SIMREADY_ASSETS_DIR>/` prefix instead of the directories'
locations, and expand them again on load, so that nothing that leaves this machine carries its
directory layout.
"""
from __future__ import annotations

import os
from pathlib import Path

PLACEHOLDER = "<ROBOLAB_DIR>/"
ROBOLAB_ROOT = os.environ.get("ROBOLAB_DIR", str(Path(__file__).resolve().parents[3] / "ext" / "RoboLab")).rstrip("/")
SIMREADY_PLACEHOLDER = "<SIMREADY_ASSETS_DIR>/"


def simready_root() -> str:
    return os.environ.get("SIMREADY_ASSETS_DIR", str(Path(__file__).resolve().parents[3] / "ext" / "simready-assets")).rstrip("/")


def portable(path):
    """The path with the RoboLab checkout written as `<ROBOLAB_DIR>` and the SimReady asset cache
    as `<SIMREADY_ASSETS_DIR>`."""
    if isinstance(path, str):
        if path.startswith(ROBOLAB_ROOT + "/"):
            return PLACEHOLDER + path[len(ROBOLAB_ROOT) + 1:]
        root = simready_root()
        if path.startswith(root + "/"):
            return SIMREADY_PLACEHOLDER + path[len(root) + 1:]
    return path


def expand(path):
    """The path with `<ROBOLAB_DIR>` and `<SIMREADY_ASSETS_DIR>` replaced by the directories."""
    if isinstance(path, str):
        if path.startswith(PLACEHOLDER):
            return ROBOLAB_ROOT + "/" + path[len(PLACEHOLDER):]
        if path.startswith(SIMREADY_PLACEHOLDER):
            return simready_root() + "/" + path[len(SIMREADY_PLACEHOLDER):]
    return path
