# OMX twin-arm teleoperation

Leader-follower teleoperation and data recording for two ROBOTIS OpenManipulator-X
arms. This is the hardware-side collection code for the Neural Actuation Dataset
(NAD): a human moves the torque-off leader arm, the follower mirrors it, and the
follower's motor telemetry is logged to CSV / published over ROS.

## Hardware

- 2x ROBOTIS OpenManipulator-X (leader + follower)
- 5x Dynamixel XM430-W350 per arm, Protocol 2.0 @ 1 Mbps
- Motor IDs: 11 (base), 12 (shoulder), 13 (elbow), 14 (wrist), 15 (gripper)
- Leader U2D2 on `/dev/ttyUSB0`, follower on `/dev/ttyUSB1`
- Optional: 6-axis force/torque sensor on `/dev/ttyACM0` (115200 8N1, ASCII frames
  `< fx fy fz mx my mz >` in mN, converted to N after tare)
- Optional: single-axis force gauge, 2400 bps ASCII (`finger_control/force_reader_class.py`)
- Optional: USB camera on `/dev/video1` + `ffmpeg` for synchronized video

## Setup (ROS 1 Noetic)

The code expects to live in a catkin package named `soft_hand_control`
(imports are `from soft_hand_control.msg import ...`):

- `dxl_arm.py` and `finger_control/` go under the package's `script/`
- `msg/*.msg` go under the package's `msg/` with `message_generation` enabled

Build with `catkin_make`, then `source devel/setup.bash`. Python dependencies:
`dynamixel-sdk`, `pyserial`, `numpy`, `matplotlib`. `roscore` must be running;
the script registers node `arm_control_node` at import time.

## Publisher (single arm)

```bash
python3 dxl_arm.py            # arm on /dev/ttyUSB0, motors disabled (free-drive)
python3 dxl_arm.py --enabled  # keep motors torqued on
python3 dxl_arm.py --usb1     # arm on /dev/ttyUSB1
```

Press Enter at the prompt after the motors are disabled. The node then publishes
`soft_hand_control/MotorMonitorNoLength` on `/dxl_arm/monitor` at 100 Hz (nominal;
bounded by the Dynamixel bulk read in practice). Message fields: `motorsPos`,
`current`, `motorsVel`, `coilsTemp`, `pwm`, `motorsVolts`, `aperture`, `force`,
`relative_time`.

## Twin teleop with NAD CSV recording

```bash
python3 dxl_arm.py --twin-csv               # CSV + synchronized video (qhd)
python3 dxl_arm.py --twin-csv --no-video
python3 dxl_arm.py --twin-csv --res=1080p   # 4k | qhd | 1080p | 720p
```

Runs `twin_motion()`: leader (ttyUSB0) is torque-off and moved by hand, follower
(ttyUSB1) mirrors it at 100 Hz in extended position mode (mode 4); the gripper is
mirrored with a -0.35 rad calibration offset. An interactive menu selects the
trajectory category; the CSV is auto-numbered as
`dataset/trajectory_data/<category>/NNN.csv` and video as
`dataset/video_data/<category>/NNN.mp4` (trimmed so video t=0 matches CSV row 0).
Press Enter to start recording, Enter again to stop (`x` also disables the follower
motors). Categories 3-5 auto-connect the 6-axis force sensor (5 s tare, keep it
unloaded) and log `force_x/y/z`.

Before logging, `record_data()` writes `PID_CONFIG` to the follower motors
(Kp=800 on all joints; Kd=1000/800/500/300/0 for base/shoulder/elbow/wrist/gripper).
Temperature guard: warning at 55 C, auto-disable at 60 C.

## Real-time teleop publisher

```bash
python3 dxl_arm.py --twin-publish
```

Runs `twin_motion_publish()`: follower mirrors the leader and publishes its
`MotorMonitorNoLength` to `/dxl_arm/monitor` at 62.5 Hz (matches the simulation
step of 0.016 s). No CSV is written.

## CSV format (NAD)

44 columns:

```
timestamp,
pos1-5, aperture,
goal_pos1-5, goal_aperture,
current1-5, vel1-5, temp1-5, pwm1-5, volts1-5,
force_x, force_y, force_z,
force_gripper_x, force_gripper_y, force_gripper_z
```

- `pos` rad, `vel` rad/s, `current` mA, `temp` C, `volts` V; `timestamp` is seconds
  since recording start
- `goal_posN`: leader position command sent to the follower
- `aperture`: gripper opening width computed from `pos5`
- `goal_aperture`: gripper opening width computed from `goal_pos5`
- `force_x/y/z`: 6-axis sensor reading in N, mapped as `force_x = -sensor_y`,
  `force_y = -sensor_x`, `force_z = -sensor_z`; `-999` means no reading
- `force_gripper_x/y/z`: payload weight label if set, else 0

## Force gauge (single-axis)

`ForceGaugeReader` in `finger_control/force_reader_class.py` reads a 2400 bps ASCII
gauge (default port `/dev/ttyUSB1`, override with `--port`). `force_gauge_run()` in
`dxl_arm.py` runs collection with the gauge polled in a background thread; the scalar
reading is projected along `force_vector` into the `force_x/y/z` columns.

## Files

```
dxl_arm.py                            # DXL_ARM class + publish/twin entry points
finger_control/finger_control.py      # FingerControlDXL: low-level Dynamixel I/O
finger_control/hand_control.py        # HandControlDXL: multi-motor group reads
finger_control/force_reader_class.py  # ForceGaugeReader: serial force gauge
msg/MotorMonitorNoLength.msg          # published on /dxl_arm/monitor
msg/MotorMonitor.msg                  # internal motor status container
msg/FingerMeasure.msg, msg/MotorPosTraj.msg, msg/MotorPosCmd.msg
```
