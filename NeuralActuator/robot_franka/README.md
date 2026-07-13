# Franka Emika Panda - MuJoCo Description

MuJoCo (MJCF) model of the Franka Emika Panda arm, adapted from
[google-deepmind/mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie)
(Apache-2.0, see [LICENSE](LICENSE)).

## Files

- `panda.xml`: the robot model, with the stock actuator definitions from the
  upstream menagerie model. Training and evaluation write the network's
  per-joint commands to these actuator inputs; the finger joint positions are
  set directly instead of driving the tendon gripper actuator.
- `scene.xml`: the robot on a ground plane with a light, used for training and
  evaluation.
