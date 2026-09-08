# serial_hand: the AmazingHand with plain hinge joints

The upstream AmazingHand model is a faithful export of the CAD assembly: two servos per finger, two rods with
ball joints, a small four-bar linkage that folds the two finger segments together, and 20 loop-closing
constraints to hold it all together. That is the right model for checking the mechanism and an awkward one for
planners, learning or retargeting. This directory builds an equivalent hand with three ordinary hinges per finger
and works out everything needed to move between the two: how the fingertip follows the knuckle, how servo angles
turn into joint angles, and which servo angles produce a given joint target.

    ah_serial/
      linkage.py        read the upstream model, find the parts of each finger, extract the geometry at the zero
                        pose, and sample the linkage over a grid of servo angles
      identify.py       fit the couplings and the servo-to-joint maps, detect the crank dead centre, collect statistics
      build_mjcf.py     write the simplified model file, reusing the upstream meshes and inertial values
      kinematics.py     runtime code: forward kinematics in numpy, servo angles to joint angles and back
      validate.py       compare the simplified model with the linkage samples, check the coupling under dynamics
      render.py         offscreen rendering, text overlays, video writing
    build_serial_model.py   the whole pipeline; about 70 s per hand
    compare_motion.py       figures/ and renders/
    make_video.py           video/linkage_vs_serial_<side>.mp4 (linkage on the left, simplified hand on the right)
    models/AH_Right, AH_Left   serial_hand.xml and scene.xml; meshes are referenced relative to the upstream assets folder
    params/                 identified_<side>.json (geometry, fitted coefficients, ranges, checks); samples_<side>.npz (raw samples)

## Which parts are kept

The upstream export names parts after the first CAD component it met, so the names say little about function.
For finger n, the parts that matter are:

| part | upstream name (right hand, finger 1) | what it is |
|---|---|---|
| G | `std00333_plast_tcb_torx_2_5x8…` | the pivot piece at the base of the finger; hinged to the palm and to the proximal segment, a two-axis pivot |
| P | `rotule_ball` (carries the "link" mesh) | the crank of the four-bar; the two rods pull on it; hinged to G |
| D | `parallel_pin_2_x_10…` (distal mesh, `tip` site) | the fingertip segment; it connects P to L and so acts as the coupler of the four-bar |
| L | `parallel_pin_2_x_16…` (proximal mesh) | the proximal segment; hinged to G at the knuckle and to D at the fingertip joint |

The simplified chain keeps G, L and D with their hinge axes:

    palm  -f{n}_abd (sideways, axis of the G-to-palm hinge)->  f{n}_gimbal
          -f{n}_mcp (knuckle, axis of the L-to-G hinge)->       f{n}_proximal
          -f{n}_pip (fingertip, axis of the D-to-L hinge)->     f{n}_distal

`f{n}_pip` has no actuator; a joint equality with a fourth-order polynomial ties it to `f{n}_mcp`. P, the rods and
the servo horns are left out of the model and only enter the servo conversion. The three kept bodies keep their
CAD coordinate frames, so their `<geom>` and `<inertial>` entries are copied from `robot.xml` unchanged. The palm,
the servos and the finger frames are kept as well, which is why the servo output shafts look bare.

Sign conventions: positive `mcp` bends the finger towards the palm, `pip` and the crank angle increase together
with `mcp`, and positive `abd` moves the fingertip towards the thumb (towards the index finger for the thumb
itself). The left hand goes through the same code; its sideways axes on fingers 1 to 3 come out mirrored. The
zero pose is the upstream `zero` keyframe, a half-curled resting pose, which is why the fingers can extend
further from it than they can bend.

## How the numbers were obtained

`LinkageHand.sample_motor_grid` commands the upstream position servos to every pair on a 5° grid of servo angles
(37 × 37 = 1369 pairs, the same command on all four fingers) with gravity and joint friction switched off. It
ramps up from the zero keyframe so the mechanism stays in the assembly it has in the CAD, waits for the servos to
settle, and records the pose of every part. For each sample, the rotation between two connected parts is split
into a turn about the hinge axis and whatever is left over. The leftover never exceeds 0.005°, which is the
evidence that a chain of three hinges describes the linkage motion exactly.

From those samples:

- `pip = poly4(mcp)` and `P = poly7(mcp)`, fitted by least squares, plus the inverse of the second. MuJoCo's joint
  equality takes at most a fourth-order polynomial, hence the order for `pip`.
- servos to `(abd, mcp)` as a sixth-order polynomial in two variables, and its inverse as a rough reference.
- the two crank-and-rod chains from geometry alone: servo axis, crank radius (5.00 mm), rod length (33.00 mm)
  and the position of the ball socket on P. Given `(abd, P)`, each chain gives one equation of the form
  `A cos θ + B sin θ = C` for its servo angle, which has a closed-form solution; the other direction is a 2 × 2
  Newton iteration started from the polynomial.

Each of those equations has two solutions, mirror images about the crank's dead centre, the position where the rod
lines up with the crank and the crank cannot push the finger any further. Between 87° and 90° of servo travel the
crank reaches that point and the finger folds back, so one pose in that narrow band corresponds to two servo
angles. Samples past a dead centre (57 of 1369 per finger) are excluded from the joint ranges and from the
inverse fits; the forward map is smooth through the dead centre and uses all samples. Extension is limited by
the dead centre, flexion by the ±90° servo range. Finger 2 loses another 30 samples close to the dead centre,
where the inverse is too sensitive to be checked; its forward map and ranges are unaffected.

Identified values, the same for all fingers and both hands:

| four-bar link lengths | ground 6.00, crank 50.50, coupler 6.00, follower 52.00 mm; the four axes are parallel |
|---|---|
| crank-and-rod chains | crank radius 5.00 mm, rod length 33.00 mm, both servo axes parallel |
| two-axis pivot | the sideways and knuckle axes meet (0.000 mm apart) at 90° |
| joint ranges | sideways ±38.7°, knuckle −61.6° to +44.2°, fingertip −63.6° to +37.6° |
| pip(mcp) polynomial coefficients (radians, lowest order first) | 3.4e-5, 0.9901, −0.0448, −0.1038, −0.0912; residual 0.016° rms, 0.08° max |
| linear approximation | mcp ≈ 0.327 (θ1 − θ2), abd ≈ 0.246 (θ1 + θ2); 3.6° / 2.3° rms error |

## Checks

Fingertip pose of the simplified model against the linkage, over the 1312 valid samples per finger (right hand;
the left hand gives the same numbers):

| joint angles obtained from | position error rms / max | orientation error rms / max |
|---|---|---|
| the linkage samples directly | 0.002 / 0.008 mm | 0.001° / 0.005° |
| servo angles through the crank-and-rod equations | 0.011 / 0.08 mm | 0.017° / 0.10° |
| servo angles through the polynomial only | 0.095 / 0.51 mm | 0.09° / 0.48° |

Also checked: joint angles to servo angles, closed form, agrees with the commanded angles to 0.05° rms; the numpy
forward kinematics in `kinematics.py` matches MuJoCo's own forward kinematics of the generated model to
0.0003 mm; stepping the simplified model with its position servos keeps the fingertip coupling error at 0.000°;
the upstream inverse-kinematics demo runs on the simplified scene (1001 solver steps in 1.6 s, coupling violated
by at most 0.22°); the 16 s video trajectory stays below 0.11 mm of fingertip error throughout.

`figures/coupling_<side>.png` shows pip and P against mcp with the polynomial and its residual;
`figures/motor_map_<side>.png` the servo-to-joint maps, with the folded-back region hatched;
`figures/tip_error_<side>.png` the error distributions for the three ways of obtaining the joints;
`figures/linear_baseline_<side>.png` why a linear approximation is not good enough.
`renders/compare_<side>.png` puts both models side by side at six poses.

## Usage

    cd serial_hand
    source ../AmazingHand/Demo/.venv/bin/activate
    export MUJOCO_GL=egl
    python build_serial_model.py --side both     # redo everything; --quick uses a 15° grid
    python compare_motion.py --side both
    python make_video.py --side right

From Python (radians throughout):

    from ah_serial.kinematics import SerialHandKinematics
    K = SerialHandKinematics.from_json("params/identified_right.json")
    q = K.motors_to_joints(theta8)          # 8 servo angles [f1m1, f1m2, ..., f4m2] -> (4, 3) [abd, mcp, pip]
    theta8 = K.joints_to_motors(q[:, :2])   # closed form, ready for the real hand
    qpos = K.qpos_from_motors(theta8)       # qpos of the simplified model: f1_abd, f1_mcp, f1_pip, f2_abd, ...
    K.in_motor_workspace(q[:, :2])          # per finger: reachable within ±90° of servo travel?

Model file: joints `f{n}_abd`, `f{n}_mcp`, `f{n}_pip`; eight position actuators `f{n}_abd` and `f{n}_mcp` with
kp = 50 like upstream; keyframe `zero` is all zeros. `scene.xml` keeps the upstream floor, lights and the four
fingertip target markers, so `mj_mink_right.py` runs on it after changing the scene path.

## Limits

Geometry and joint relations only: inertial values are copied for the three kept bodies, the rods and cranks (a
few grams) are ignored, and nothing was done about servo dynamics. No collision geometry, same as upstream. The
joint ranges are the bounding box of the reachable region; use `in_motor_workspace` before planning to a corner of
it. Poses past a dead centre cannot be represented.
