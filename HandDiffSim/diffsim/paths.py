"""Locations of the hand models and identified parameters, relative to amazing_hand_sim/."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                                   # amazing_hand_sim/
SERIAL_DIR = ROOT / "serial_hand"
UPSTREAM = ROOT / "AmazingHand"


def serial_model_xml(side: str = "right") -> Path:
    return SERIAL_DIR / "models" / f"AH_{side.capitalize()}" / "serial_hand.xml"


def identified_params(side: str = "right") -> Path:
    return SERIAL_DIR / "params" / f"identified_{side}.json"


def linkage_scene_xml(side: str = "right") -> Path:
    return UPSTREAM / "Demo/AHSimulation/AHSimulation" / f"AH_{side.capitalize()}" / "mjcf" / "scene.xml"
