"""DSAD: Dynamic Soft Anisotropic Diagrams for reduced-order video representation.

A GoP is N SAD sites (sad_appearance_v3) moved by a reduced-order model over
M anchors,

    Δ(x, t) = Σ_j φ_j(x) a_j(t),    M << N,

with anchor weights φ_j(x) (MotionSAD) and knot-based temporal coefficients
a_j(t) (Temporal1DSAD), combined in DeformModelSAD. train_sad_video.py fits a
GoP, export.py writes its stored form and decode.py decodes it.
"""
from .motion_sad import MotionSAD
from .temporal_1d_sad import Temporal1DSAD
from .deformation_core import DeformModelSAD, SADDeformCore
from . import losses_motion

__all__ = [
    "MotionSAD",
    "Temporal1DSAD",
    "DeformModelSAD",
    "SADDeformCore",
    "losses_motion",
]
