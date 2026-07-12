# Neural Actuation Dataset (NAD)

The dataset ships with this repository under `data/` — no separate download. It covers
two platforms:

- **OpenManipulator-X** (Dynamixel XM430): 35 tasks, 350 trajectories, ~319k frames
  (~90 minutes) at ~58.8 Hz.
- **SO-101** (Feetech STS3215, LeRobot ecosystem): 10 task-payload combinations,
  100 trajectories, ~66k frames (~18 minutes) at ~62.3 Hz.

Every task directory contains 8 training, 1 validation and 1 test trajectory
(`train/001.csv` ... `train/008.csv`, `validation/001.csv`, `test/001.csv`). Validation
and test trajectories are held-out repetitions of the same commanded trajectory as the
training files; the on-disk split is used as-is by all configs.

The collection hardware is listed in the [Hardware](../README.md#hardware) section of
the main README; the recording pipeline (teleop rig, motor IDs, PID settings, force
sensor tare) is documented in [teleop/README.md](../teleop/README.md).

## Directory layout

```
data/
  force_unlabeled/            # OMX free motion, 10 tasks
    backward_forward/{train,validation,test}/*.csv
    circular_cw/  circular_ccw/  go_up_and_stay_still/
    joint_sweep_motor11/ ... joint_sweep_motor15/
    pick_place_empty/
  force_sensor/               # OMX end-effector pushes, 12 tasks
    force_x_plus/  force_x_minus/  force_y_plus/  ...  force_z_minus/
    force_x_plus_ref/  ...  force_z_minus_ref/        # matched no-contact references
  weight/                     # OMX payload tasks, 9 tasks
    go_up_and_stay_still_no_force/
    go_up_and_stay_still_with_object_{200g,300g,400g}/
    pick_place_no_force/
    pick_place_object_{200g,300g,400g,500g}/
  motor_condition/            # OMX normal vs degraded joint 3, 4 tasks
    pick_place_empty/  pick_place_empty_degrade/
    pick_place_object_200g/  pick_place_object_200g_degrade/
  so101/                      # SO-101, task x payload
    go_up_and_stay_still/{empty,200g,300g,400g,500g}/{train,validation,test}/*.csv
    pick_and_place/{empty,200g,300g,400g,500g}/{train,validation,test}/*.csv
```

## OpenManipulator-X CSVs

44 columns per row, one row per telemetry sample (~58.8 Hz; the configs assume a
17 ms step, `data_dt: 0.017`). The follower arm mirrors a hand-moved leader; `goal_*`
columns are the leader commands sent to the follower, everything else is follower
telemetry.

| Columns | Unit | Description |
|---|---|---|
| `timestamp` | s | Time since recording start |
| `pos1`-`pos5` | rad | Joint positions (joint 5 is the gripper motor) |
| `aperture` | mm | Gripper opening width, derived from `pos5` |
| `goal_pos1`-`goal_pos5` | rad | Commanded joint positions (leader) |
| `goal_aperture` | mm | Commanded gripper opening, derived from `goal_pos5` |
| `current1`-`current5` | mA | Motor currents |
| `vel1`-`vel5` | rad/s | Joint velocities |
| `temp1`-`temp5` | C | Motor temperatures |
| `pwm1`-`pwm5` | counts | Signed XM430 PWM duty register |
| `volts1`-`volts5` | V | Bus voltages |
| `force_x`, `force_y`, `force_z` | N | External force at the end effector (see below) |
| `force_gripper_x/y/z` | N | Reserved; all zero in the released data |

The force channels use `-999` as a sentinel for frames without a force reading. What
they contain depends on the subset:

| Subset | Force channels |
|---|---|
| `force_unlabeled` | `-999` throughout (no sensor attached) |
| `force_sensor`, push directions | 6-axis F/T sensor reading on all three axes |
| `force_sensor`, `*_ref` | `-999` throughout (no-contact reference runs) |
| `weight` | `force_z` = payload weight (e.g. -4.9 N for 500 g) while the object is held, `-999` otherwise; `force_x/y` always `-999` |
| `motor_condition` | as `weight`: the 200 g tasks carry the `force_z` payload label, the empty tasks are `-999` throughout |

The OMX loaders (`evaluate_actuator.py`, used by training and evaluation) map `-999`
to 0 N at load time, i.e. frames without a reading are treated as zero external force.

The training features use `goal_pos`, `pos`, `aperture`, `current`, `vel`, `volts`,
`temp` and the goal columns; `pwm` and `force_gripper_*` are recorded but unused.

Benchmark mapping:

| Subset | Benchmark | Config |
|---|---|---|
| `force_unlabeled`, 8 arm-only tasks | Table 1, rows 1-8 | `configs/no_load_no_gripper.yaml` |
| `force_unlabeled/joint_sweep_motor15`, `pick_place_empty` | Table 1, rows 9-10 | `configs/no_load_with_gripper.yaml` |
| `force_sensor`, all 12 tasks | Table 2 | `configs/force_sensor.yaml` |
| `weight`, all 9 tasks | Table 3 | `configs/weight_all.yaml` |
| `weight`, 4 go-up-and-stay tasks | Table 3 subset | `configs/lift_hold.yaml` |
| `weight`, 5 pick-and-place tasks | Table 3 subset | `configs/pick_place.yaml` |
| `motor_condition`, all 4 tasks | Table 4 | `configs/motor_condition.yaml` |

## SO-101 CSVs

46 columns per row at ~62.3 Hz (`data_dt: 0.01605` in the config). Six joints; the
jaw is joint 6, so there are no separate gripper/aperture columns. Servo telemetry is
kept in raw STS3215 register units.

| Columns | Unit | Description |
|---|---|---|
| `timestamp` | s | Time since recording start |
| `pos1`-`pos6` | rad | Joint positions |
| `goal_pos1`-`goal_pos6` | rad | Commanded joint positions (leader) |
| `current1`-`current6` | counts | Current register (unused by the released configs) |
| `vel1`-`vel6` | steps/s | Velocity in encoder steps per second (4096 steps/rev) |
| `load1`-`load6` | counts | Signed load register; this is the "current" feature the released configs read (`current_source: load`) |
| `temp1`-`temp6` | C | Motor temperatures |
| `volts1`-`volts6` | dV | Bus voltage in decivolts (e.g. 122 = 12.2 V) |
| `force_x`, `force_y`, `force_z` | N | Weight-derived force label (see below) |

The SO-101 rig has no end-effector force sensor. Force labels are derived from the
known payload weight: `force_z` equals the payload weight (e.g. -2.94 N for 300 g)
while the object is held and `-999` otherwise; `force_x` and `force_y` are `-999`
throughout. Unlike the OMX pipeline, the SO-101 config sets `mask_invalid_force: true`,
which excludes `-999` frames from the force loss instead of zeroing them.

All ten task-payload combinations feed the SO-101 weight benchmark
(`configs/so101_weight.yaml`); the paper's protocol uses the six 300-500 g
combinations, see the SO-101 section of the main README.

## License

The dataset is released under the same MIT license as the code (see
[LICENSE](../LICENSE) at the repository root). The robot models under `robot/` and
`robot_so101/` are third-party assets under their own Apache-2.0 licenses and are not
part of the dataset.

## Collection videos

Camera recordings of the collection sessions (430 MP4s, one per released trajectory;
the two no-load reference tasks were not recorded) are hosted with the Hugging Face
mirror of this dataset, under `videos/`, at
https://huggingface.co/datasets/frankzydou/NAD — they are not included in the GitHub
repository. OMX clips are 1706x1440 at 24 fps, SO-101 clips 1280x1440 at 30 fps; the
videos are session recordings and are not frame-synchronized to the CSVs.
