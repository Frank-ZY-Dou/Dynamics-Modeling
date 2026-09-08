"""ah_serial: a simplified serial-joint MuJoCo model of the Pollen Robotics AmazingHand.

Pipeline
--------
linkage.py   : probe the official closed-chain MJCF, extract zero-pose geometry, sample its motion
identify.py  : fit joint couplings / motor maps, analytic 2-RSS closure model (parameter identification)
build_mjcf.py: generate the serial MJCF (palm -> abd -> mcp -> pip per finger) reusing the CAD meshes
kinematics.py: runtime API (numpy FK, motors <-> joints) from the identified JSON parameters
"""
from .linkage import LinkageHand, FingerStructure, FingerGeometry, MotionSamples  # noqa: F401
from .kinematics import SerialHandKinematics  # noqa: F401
