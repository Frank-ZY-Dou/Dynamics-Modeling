# Hardware observations

## Gesture targets, 2026-09-17

Hardware: left Wuji Hand 2 Beta 2. Software: Python 3.10.12,
`wuji-sdk==2026.8.31`, NumPy 2.2.6, MuJoCo 3.13.0, and
OpenCV headless 5.0.0.93.

The original gesture controller completed open palm, pointing, peace, thumbs up,
I love you, and shaka. Targets in `gestures.py` preserve those joint angles.
The run used `kp=3`, `kd=0.05`, a 0.5 A current cap, and smooth commands limited to
0.2 rad/s. Maximum errors across the 20 joints at the six holds ranged from
0.036 to 0.063 rad. The run returned open and confirmed all motors disabled.

These are observations of one hand and configuration. They do not establish
right-hand targets or behavior on different hardware revisions. The reorganized
package is covered by offline tests; it has not been enabled on a physical hand
as part of this publication.

## Reporting a hardware run

Record the hardware revision, handedness, firmware and SDK versions, model
revision, control settings, requested poses, and whether disable and restoration
were confirmed. Report the first fault or tracking failure, including its stage.
Keep the original timestamped recordings locally so command timing can be checked.
