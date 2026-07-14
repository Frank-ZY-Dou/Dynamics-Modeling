# Teleoperation and data collection

Leader–follower teleoperation and data-recording code used to collect the Neural
Actuation Dataset (NAD). A human moves the torque-off **leader** arm, the **follower**
mirrors it, and the follower's motor telemetry is logged to CSV alongside synchronized
video. The same two-phase workflow is used on both platforms: teleoperate freely, then
press ENTER to start a synchronized CSV + video recording.

| Platform | Servos | Code | Guide |
|---|---|---|---|
| **OpenManipulator-X** (ROS 1) | 5× Dynamixel XM430-W350 | [`omx/`](omx/) | [omx/README.md](omx/README.md) |
| **SO-101** (LeRobot) | 6× Feetech STS3215 | [`so101/`](so101/) | [so101/README.md](so101/README.md) |

Both produce CSVs in the same layout, so the SO-101 recordings drop straight into the
NAD schema in [docs/dataset.md](../docs/dataset.md).

## OpenManipulator-X (`omx/`)

ROS 1 (Noetic) leader/follower over Dynamixel SDK: `dxl_arm.py` is the per-arm
publisher/recorder, `finger_control/` reads the gripper force gauge, and `msg/` holds
the custom ROS messages. Optional 6-axis force/torque sensor and USB camera. See
[omx/README.md](omx/README.md) for the catkin package layout, wiring, and motor IDs.

## SO-101 (`so101/`)

LeRobot-based leader/follower over the Feetech serial bus: `record_teleop_data.py` is
the recorder, with post-processing scripts for force labelling (`label_force_z.py`,
`lift_and_hold_set_force_z.py`, `add_force_columns.py`), unit conversion
(`convert_pos_to_rad.py`), dataset splitting (`split_dataset.py`) and stats
(`dataset_stats.py`). Requires a LeRobot install with the SO-101 leader/follower
classes. See [so101/README.md](so101/README.md).

## Hardware sourcing

The component list with purchase links is in the
[Hardware and Data Collection](../README.md#hardware-and-data-collection) section of the
main README.
