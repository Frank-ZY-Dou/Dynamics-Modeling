"""Asset paths inside and outside the RoboLab checkout.

Layouts, certificates and export manifests write RoboLab asset paths with a `<ROBOLAB_DIR>/`
prefix instead of the checkout's location, and expand it again on load, so that nothing that
leaves this machine carries its directory layout.
"""
from __future__ import annotations

import os
from pathlib import Path

PLACEHOLDER = "<ROBOLAB_DIR>/"
ROBOLAB_ROOT = os.environ.get("ROBOLAB_DIR", str(Path(__file__).resolve().parents[3] / "ext" / "RoboLab")).rstrip("/")


def portable(path):
    """The path with the RoboLab checkout written as `<ROBOLAB_DIR>`."""
    if isinstance(path, str) and path.startswith(ROBOLAB_ROOT + "/"):
        return PLACEHOLDER + path[len(ROBOLAB_ROOT) + 1:]
    return path


def expand(path):
    """The path with `<ROBOLAB_DIR>` replaced by the checkout."""
    if isinstance(path, str) and path.startswith(PLACEHOLDER):
        return ROBOLAB_ROOT + "/" + path[len(PLACEHOLDER):]
    return path
