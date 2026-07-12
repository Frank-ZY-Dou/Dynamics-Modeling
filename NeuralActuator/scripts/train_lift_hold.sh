#!/bin/bash
# usage: bash scripts/train_lift_hold.sh [gpu_id]
GPU=${1:-0}
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
CUDA_VISIBLE_DEVICES=$GPU python train_actuator_diffsim.py \
    --train_config configs/lift_hold.yaml \
    --seed 0 \
    --log_dir outputs/logs_lift_hold \
    --model_out outputs/lift_hold_params.pkl
