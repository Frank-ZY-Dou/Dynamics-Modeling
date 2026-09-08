"""Differentiable simulation of the simplified AmazingHand on MuJoCo Warp.

backend.py           batched control step of the hand as a torch.autograd.Function
kinematics_torch.py  differentiable fingertip forward kinematics (torch)
servo.py             position-servo torque law and its inverse (torque -> command)
models.py            torque-surrogate networks (MLP, Transformer)
synth_data.py        reference trajectories from the original linkage model
train_actuator.py    learn a torque surrogate by backpropagating pose error through the simulator
"""
