# Dynamics-Modeling

Research code and open-source projects on dynamics modeling and simulation.

## Papers

- [**NeuralActuator**](NeuralActuator/) — Neural Actuation Modeling for Robot Dynamics and External Force Perception.
  Robotics: Science and Systems (RSS) 2026.<!-- **Finalist for the Outstanding Student Paper and Outstanding Paper Awards.** -->
- [**RigidFormer**](RigidFormer/) — Learning Rigid Dynamics with Transformers.
- [**S4R**](Penetration_Solving/) — Scaling for Rigid-Body Interpenetration Resolution.
  SIGGRAPH Asia 2026 (ACM Transactions on Graphics). An agentic, language-driven front end that repairs generated RoboLab and RoboCasa
  scenes with S4R is in [SimReady Gate](SimReady_Gate/).

## Projects

- [**Differentiable SuperDex**](https://github.com/Frank-ZY-Dou/differentiable-superdex) — A fork of Meta's
  [Project SuperDex](https://github.com/facebookresearch/project_superdex) whose contact-first physics engine
  is made differentiable: the discrete adjoint through frictional contact for rigid, articulated, soft and rod
  bodies, a PyTorch bridge for policy and parameter learning, and manipulation demos solved by gradient
  descent through contact.
- [**HandDiffSim**](HandDiffSim/) — Differentiable simulation of the Pollen Robotics AmazingHand on MuJoCo Warp.
- [**SimReady Gate**](SimReady_Gate/) — Language-driven, verifier-in-the-loop repair that turns generated tabletop
  scenes into penetration-free, simulation-ready ones: a request becomes a typed constraint program, S4R repairs
  the scene under it in scale-space, and a mesh-level evaluator plus a MuJoCo settle certify the result. Reads
  RoboLab (USD) and RoboCasa (MJCF) scenes directly; measured on their assets.
