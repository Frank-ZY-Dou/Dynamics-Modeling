"""Import shim: adds the NeuralActuator package dir to sys.path and re-exports
load_dataset, sample_valid_indices, validate_mujoco_joint_limits, create_model.
"""
from __future__ import annotations

import os
import sys

PUBLIC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
assert os.path.isdir(PUBLIC_DIR), f"NeuralActuator dir missing: {PUBLIC_DIR}"
if PUBLIC_DIR not in sys.path:
    sys.path.insert(0, PUBLIC_DIR)

from train_actuator_diffsim import (  # noqa: E402
    load_dataset,
    sample_valid_indices,
    validate_mujoco_joint_limits,
)
from models import create_model  # noqa: E402

__all__ = ["PUBLIC_DIR", "load_dataset", "sample_valid_indices",
           "validate_mujoco_joint_limits", "create_model"]
