# SO101 (LeRobot) Data Collection

Leader–follower teleoperation data collection for the **SO101** arm (Feetech
STS3215 servos), producing CSV + video in a format compatible with the
OpenManipulator pipeline in `../omx/`.

These scripts build on a LeRobot fork. They depend on the custom
`so_follower` / `so_leader` classes in that fork; install it (editable) in a
`lerobot` conda env so `import lerobot` works, or point to its sources with
`export LEROBOT_SRC=/path/to/lerobot/src`.

---

## Hardware

| | Leader | Follower |
|---|---|---|
| Port | `/dev/ttyACM0` | `/dev/ttyACM1` |
| ID | `my_awesome_leader_arm` | `my_awesome_follower_arm` |
| Servos | 6× Feetech STS3215 | 6× Feetech STS3215 |

Motors (in order): `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper`
Camera: OpenCV device `/dev/video0` (default QHD 2560×1440 @ 30 fps).
Sensor logging rate: ~62.5 Hz (16 ms loop).

---

## 1. Record a trajectory

```bash
conda activate lerobot
cd teleop/so101

python record_teleop_data.py                 # default: QHD video + CSV
python record_teleop_data.py --no-video      # CSV only
python record_teleop_data.py --duration 60   # auto-stop after 60 s
python record_teleop_data.py --zoom 2.0      # 2× digital zoom
python record_teleop_data.py --res 1080p     # resolution preset: 4k / qhd / 1080p / 720p / WxH
```

Optional overrides: `--follower-port`, `--leader-port`, `--camera-index`,
`--video-fps`, `--fourcc`.

**Two-phase workflow** (same as OpenManipulator `--twin-csv`):

1. **Teleop phase** — follower tracks leader, **nothing saved**. Move the arm
   freely / position it. Press **ENTER** to begin recording.
2. **Recording phase** — CSV + video start synchronized. Press **ENTER** or
   **Ctrl+C** to stop and save.

At launch you pick a category from a menu, then a file number
(auto-incremented). Data is saved under `dataset/` next to the script:

```
dataset/
├── trajectory_data/
│   ├── lift_and_hold/{200g,300g,400g,500g,empty}/   001.csv, 002.csv ...
│   └── pick_and_place/{200g,300g,400g,500g,empty}/  001.csv, 002.csv ...
└── video_data/
    └── (mirror structure)                            001.mp4, 002.mp4 ...
```

### CSV columns

```
timestamp,
pos1-6, goal_pos1-6, current1-6, vel1-6, load1-6, temp1-6, volts1-6,
force_x, force_y, force_z
```

- `pos*` / `goal_pos*` are in **radians** (converted on the fly from STS3215 degrees).
- `force_*` default to `-999` (no F/T sensor) and are filled in during labeling.
- Compared to the OpenManipulator format: 6 motors instead of 5, includes
  `goal_pos` and `load`, and has no `aperture` / `pwm` columns.

---

## 2. Post-processing pipeline

Run in order after recording a batch. All scripts accept a single file, a glob,
or a folder (with or without `train/validation/test` subfolders).

```bash
# (optional) legacy CSVs stored as raw ticks → radians, using follower calibration
python convert_pos_to_rad.py dataset/trajectory_data/lift_and_hold/200g

# ensure force_x/y/z columns exist (default -999)
python add_force_columns.py dataset/trajectory_data/pick_and_place/200g/*.csv

# --- label force_z ---
# lift_and_hold: held weight is constant → set force_z = -m*g for the whole folder
python lift_and_hold_set_force_z.py dataset/trajectory_data/lift_and_hold/200g -1.962
python lift_and_hold_set_force_z.py dataset/trajectory_data/lift_and_hold/500g -4.905

# pick_and_place: label force_z only during the carry interval (interactive
# prompts for start/end timestamp and value)
python label_force_z.py dataset/trajectory_data/pick_and_place/200g/001.csv

# crop videos (keep center half, mute audio); original kept as *_original.mp4
python crop_videos.py dataset/video_data/lift_and_hold/200g

# split into train/validation/test, synced between trajectory_data and video_data
python split_dataset.py dataset/trajectory_data/lift_and_hold/200g --seed 42

# report frame counts / durations
python dataset_stats.py dataset/trajectory_data/lift_and_hold
```

**force_z reference** (`-m·g`, g = 9.81 m/s²):

| Weight | force_z (N) |
|--------|-------------|
| 200 g  | -1.962 |
| 300 g  | -2.943 |
| 400 g  | -3.924 |
| 500 g  | -4.905 |

---

## Troubleshooting

- **`ImportError: lerobot...`** — activate the env (`conda activate lerobot`) or
  set `export LEROBOT_SRC=/path/to/lerobot/src`.
- **USB permission** — `sudo chmod 666 /dev/ttyACM0 /dev/ttyACM1` or add yourself
  to `dialout` (`sudo usermod -a -G dialout $USER`).
- **Wrong ports** — check `ls /dev/ttyACM*`; pass `--leader-port` / `--follower-port`.
- **Camera fails to open** — verify `--camera-index` (`ls /dev/video*`); lower
  `--res` (e.g. `720p`) if the camera can't sustain high-res 30 fps.
