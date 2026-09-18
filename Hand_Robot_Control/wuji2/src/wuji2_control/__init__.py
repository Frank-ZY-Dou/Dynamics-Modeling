"""Wuji Hand 2 communication and supervised joint control."""

from .config import ConnectionConfig, ControlConfig
from .device import Wuji2Device
from .motion import Waypoint
from .session import ControlSession

__version__ = "0.1.0"
__all__ = ["ConnectionConfig", "ControlConfig", "ControlSession", "Waypoint", "Wuji2Device"]
