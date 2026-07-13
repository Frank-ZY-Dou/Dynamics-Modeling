#!/bin/bash
# usage: bash scripts/train_franka_lift_hold_ft1.sh [gpu_id]
GPU=${1:-0}
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
CUDA_VISIBLE_DEVICES=$GPU python train_actuator_diffsim_franka.py \
    --train_config configs/franka_lift_hold_ft1.yaml \
    --log_dir outputs/logs_franka_lift_hold_ft1 \
    --model_out outputs/franka_lift_hold_ft1_params.pkl
