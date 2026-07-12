#!/bin/bash
# usage: bash scripts/train_motor_condition.sh [gpu_id]
GPU=${1:-0}
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
CUDA_VISIBLE_DEVICES=$GPU python train_actuator_diffsim.py \
    --train_config configs/motor_condition.yaml \
    --seed 0 \
    --log_dir outputs/logs_motor_condition \
    --model_out outputs/motor_condition_params.pkl
