# Dynamics-Modeling

This repo contains research code for:

- [**NeuralActuator**](NeuralActuator/) — Neural Actuation Modeling for Robot Dynamics and External Force Perception.
  Robotics: Science and Systems (RSS) 2026.<!-- **Finalist for the Outstanding Student Paper and Outstanding Paper Awards.** -->
- [**RigidFormer**](RigidFormer/) — Learning Rigid Dynamics with Transformers
- [**S4R**](Penetration_Solving/) — Scaling for Rigid-Body Interpenetration Resolution.
  SIGGRAPH Asia 2026 (ACM Transactions on Graphics).

## Differentiable simulation

- [**Differentiable SuperDex**](https://github.com/Frank-ZY-Dou/differentiable-superdex) — A fork of Meta's
  [Project SuperDex](https://github.com/facebookresearch/project_superdex) whose contact-first physics engine
  is made differentiable: the discrete adjoint through frictional contact for rigid, articulated, soft and rod
  bodies, a PyTorch bridge for policy and parameter learning, and manipulation demos solved by gradient
  descent through contact.

<p align="center">
<img src="https://github.com/Frank-ZY-Dou/differentiable-superdex/raw/main/superdex_physics/examples/media/robot_push_policy.gif" width="42%" alt="A feedback policy pushes a cube">&emsp;
<img src="https://github.com/Frank-ZY-Dou/differentiable-superdex/raw/main/superdex_physics/examples/media/robot_wuji2_grasp.gif" width="42%" alt="A Wuji Hand 2 carries a cube">
</p>
