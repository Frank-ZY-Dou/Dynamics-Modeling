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
- [**Hand_Robot_Control**](Hand_Robot_Control/) — Communication, joint control, and gesture examples for robotic hands, starting with Wuji Hand 2.
- [**SimReady Gate**](SimReady_Gate/) — Language-driven, verifier-in-the-loop repair that turns generated tabletop
  scenes into penetration-free, simulation-ready ones: a request becomes a typed constraint program, S4R repairs
  the scene under it in scale-space, and a mesh-level evaluator plus a MuJoCo settle certify the result. Reads
  RoboLab (USD) and RoboCasa (MJCF) scenes directly; measured on their assets.

## Surveys & Living Archives

- [**Frontier AI for 3D Modeling & Robotics (Visual Case Archive)**](https://mit-cdfg.github.io/Survey-AI-for-3D-modeling-Robotics/#view-gallery) — A systematic empirical horizon scan and interactive case registry cataloging 190+ showcases across parametric CAD, 3D generative scenes, physics-grounded simulations (Isaac Sim, MuJoCo, Genesis), and real-world robotic control. Structured around a three-tier reproducibility hierarchy (Rank 1 verified code/harnesses, Rank 2 interactive cloud viewers, Rank 3 demonstration media) to audit spatial reasoning, closed-loop dynamics, and simulation-to-reality transfer in frontier foundation models. Companion repository: [awesome-ai-3d-modeling-robotics](https://github.com/Frank-ZY-Dou/awesome-ai-3d-modeling-robotics).
