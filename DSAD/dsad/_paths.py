"""Paths and import setup shared by the DSAD entry points.

The submodules and the extension cache are found from this file's location.
The CUDA extensions that PyTorch compiles on first use (dsad_fast,
dsad_exact_candidates, and gsplat_cuda of the D2GV submodule) are cached in
.torch_extensions/<python>_<cuda>_<torch>/ in the DSAD directory unless
TORCH_EXTENSIONS_DIR is set. setup.sh builds them once, so that concurrent
jobs only load them.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DSAD_DIR = REPO / "dsad"
SAD_BACKEND = REPO / "references" / "SAD" / "backends" / "pytorch"
D2GV_DIR = REPO / "baselines" / "D2GV"
GSPLAT_DIR = D2GV_DIR / "gsplat"


def extensions_dir() -> Path:
    """Default JIT-extension cache: one directory per Python / CUDA / torch version."""
    import torch
    cuda = (torch.version.cuda or "cpu").replace(".", "")
    tag = (f"py{sys.version_info.major}{sys.version_info.minor}_cu{cuda}_"
           f"torch{torch.__version__.split('+')[0]}")
    return REPO / ".torch_extensions" / tag


def setup_env() -> None:
    """Set TORCH_EXTENSIONS_DIR to the DSAD directory's cache unless it is already set, and put
    the interpreter's bin directory on PATH so that the JIT builder finds its ninja."""
    if not os.environ.get("TORCH_EXTENSIONS_DIR"):
        os.environ["TORCH_EXTENSIONS_DIR"] = str(extensions_dir())
    bindir = os.path.dirname(sys.executable)
    if bindir not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = bindir + os.pathsep + os.environ.get("PATH", "")


def setup_imports() -> None:
    """Put dsad/, the D2GV submodule and its gsplat on sys.path: the trainer imports
    them as top-level modules."""
    for p in (D2GV_DIR, GSPLAT_DIR, DSAD_DIR):   # final order: dsad, gsplat, D2GV
        s = str(p)
        if s in sys.path:
            sys.path.remove(s)
        sys.path.insert(0, s)


def require_d2gv() -> None:
    """Raise ImportError with instructions if the D2GV submodule is missing. The sites
    are optimized with its Adan (optimizer.py); --appearance gaussian also runs its
    model and loss."""
    if not (D2GV_DIR / "optimizer.py").is_file():
        raise ImportError(f"the D2GV submodule is missing ({D2GV_DIR} has no optimizer.py): fetch it with "
                          "`git submodule update --init`, or run setup.sh")


def setup() -> None:
    setup_env()
    setup_imports()
