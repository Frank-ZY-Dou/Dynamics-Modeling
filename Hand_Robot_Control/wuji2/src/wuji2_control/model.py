"""Fixed-base model validation and sampled self-collision checks."""

import importlib
from numbers import Integral
from pathlib import Path

import numpy as np

from .motion import _finite_scalar, _joint_vector

SDK_JOINT_NAMES = (
    "l_thumb_cmc_flex",
    "l_thumb_cmc_abd",
    "l_thumb_mcp",
    "l_thumb_ip",
    "l_index_finger_mcp_flex",
    "l_index_finger_mcp_abd",
    "l_index_finger_pip",
    "l_index_finger_dip",
    "l_middle_finger_mcp_flex",
    "l_middle_finger_mcp_abd",
    "l_middle_finger_pip",
    "l_middle_finger_dip",
    "l_ring_finger_mcp_flex",
    "l_ring_finger_mcp_abd",
    "l_ring_finger_pip",
    "l_ring_finger_dip",
    "l_pinky_mcp_flex",
    "l_pinky_mcp_abd",
    "l_pinky_pip",
    "l_pinky_dip",
)


class JointModel:
    """Validate SDK-order poses against an explicitly supplied left-hand MJCF.

    MuJoCo is imported only when constructing this object. Both official
    fixed-base left.xml and left_with_mount.xml models are supported. Model
    joint storage order may differ; named joint qpos addresses define mapping.
    Instances reuse collision-check state and are intended for one control
    session at a time.
    """

    joint_names = SDK_JOINT_NAMES

    def __init__(
        self, path: Path, handedness="left", margin=0.05, penetration_tolerance=0.0002, samples=151
    ):
        if handedness != "left":
            raise ValueError("Only the left-hand model has been validated")
        self.path = Path(path).expanduser().resolve(strict=True)
        if not self.path.is_file():
            raise ValueError("Model path must identify an MJCF file")
        self.margin = _finite_scalar(margin, "margin")
        self.penetration_tolerance = _finite_scalar(penetration_tolerance, "penetration_tolerance")
        if self.margin < 0 or self.penetration_tolerance < 0:
            raise ValueError("margin and penetration_tolerance must be nonnegative")
        if isinstance(samples, bool) or not isinstance(samples, Integral) or samples < 3:
            raise ValueError("samples must be an integer of at least 3")
        self.samples = int(samples)
        try:
            self._mj = importlib.import_module("mujoco")
        except ImportError as exc:
            raise ImportError("MuJoCo is required to load and validate the hand model") from exc
        self._model = self._mj.MjModel.from_xml_path(str(self.path))
        model = self._model
        if model.njnt != 20 or model.nq != 20 or model.nv != 20:
            raise ValueError(
                "Model must have exactly 20 fixed-base hinge joints; extra or free-base joints are unsupported"
            )
        if np.any(model.jnt_type != self._mj.mjtJoint.mjJNT_HINGE):
            raise ValueError("Every model joint must be a hinge")
        names = tuple(
            self._mj.mj_id2name(model, self._mj.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)
        )
        if set(names) != set(SDK_JOINT_NAMES):
            missing = sorted(set(SDK_JOINT_NAMES) - set(names))
            unexpected = sorted(str(name) for name in set(names) - set(SDK_JOINT_NAMES))
            raise ValueError(
                f"Model joint names do not match the left hand: missing={missing}, unexpected={unexpected}"
            )
        ids = np.array(
            [
                self._mj.mj_name2id(model, self._mj.mjtObj.mjOBJ_JOINT, name)
                for name in SDK_JOINT_NAMES
            ],
            dtype=int,
        )
        self._qpos_addresses = model.jnt_qposadr[ids].copy()
        limits = model.jnt_range[ids].copy()
        if not np.all(model.jnt_limited[ids]) or not np.isfinite(limits).all():
            raise ValueError("Every hand joint must have finite enabled position limits")
        self._lower = limits[:, 0] + self.margin
        self._upper = limits[:, 1] - self.margin
        if np.any(self._lower >= self._upper):
            raise ValueError("Model joint limits leave no range within the requested margin")
        self._joint_limits = limits
        self._data = self._mj.MjData(model)

    @property
    def joint_limits(self) -> np.ndarray:
        """Return original model limits in SDK order as an independent copy."""
        return self._joint_limits.copy()

    def _check_limits(self, q: np.ndarray) -> None:
        outside = np.flatnonzero((q < self._lower) | (q > self._upper))
        if outside.size:
            i = int(outside[0])
            raise ValueError(
                f"{SDK_JOINT_NAMES[i]}={q[i]:.6g} is outside the margin-adjusted range "
                f"[{self._lower[i]:.6g}, {self._upper[i]:.6g}]"
            )

    def _check_collisions(self, q: np.ndarray) -> None:
        self._mj.mj_resetData(self._model, self._data)
        self._data.qpos[self._qpos_addresses] = q
        self._mj.mj_forward(self._model, self._data)
        for contact in self._data.contact:
            if contact.dist < -self.penetration_tolerance:
                bodies = [
                    self._mj.mj_id2name(
                        self._model, self._mj.mjtObj.mjOBJ_BODY, int(self._model.geom_bodyid[geom])
                    )
                    for geom in (contact.geom1, contact.geom2)
                ]
                raise ValueError(
                    f"Self-collision between {bodies[0]} and {bodies[1]}: "
                    f"penetration {-float(contact.dist):.6g} m"
                )

    def validate(self, q) -> None:
        """Reject invalid dimensions, nonfinite positions, limits, or collision."""
        vector = _joint_vector(q)
        self._check_limits(vector)
        self._check_collisions(vector)

    def preflight(self, start, end) -> None:
        """Check the whole interpolated segment at evenly spaced samples.

        A quintic time law follows this same geometric segment. Sampling does
        not guarantee detection of contact narrower than the sample spacing.
        """
        start = _joint_vector(start, "start")
        end = _joint_vector(end, "end")
        self._check_limits(start)
        self._check_limits(end)
        for i, q in enumerate(np.linspace(start, end, self.samples)):
            try:
                self._check_collisions(q)
            except ValueError as exc:
                raise ValueError(f"Trajectory sample {i}/{self.samples - 1}: {exc}") from exc
