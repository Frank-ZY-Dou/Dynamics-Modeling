# SO101 Robot - MuJoCo Description

MuJoCo (MJCF) model of the SO-101 arm, adapted from
[TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100)
(Apache-2.0, see [LICENSE](LICENSE)).

## Overview

- The robot model was generated using the [onshape-to-robot](https://github.com/Rhoban/onshape-to-robot) plugin from a CAD model designed in Onshape.
- Base collision meshes were removed due to problematic collision behavior during simulation and planning.
- Joint calibration follows the upstream "new" convention: each joint's virtual zero is set to the middle of its joint range.

## Files

- `so101_new_calib.xml`: the exported robot model.
- `so101_torque_scene.xml`: the training and evaluation scene used by this repository — same kinematics and dynamics, with the position actuators replaced by torque (motor) actuators and the wrist_flex joint range widened to cover the recorded trajectories.

## Motor Parameters

Motor properties for the STS3215 motors used in the robot are adapted from the [Open Duck Mini project](https://github.com/apirrone/Open_Duck_Mini).

## Gripper Note

In LeRobot, the gripper is represented as a **linear joint**, where `0` = fully closed and `100` = fully open. This mapping is not reflected in the MJCF files here.
