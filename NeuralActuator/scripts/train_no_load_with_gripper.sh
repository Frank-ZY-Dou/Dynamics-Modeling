#!/bin/bash
# usage: bash scripts/train_no_load_with_gripper.sh [gpu_id]
GPU=${1:-0}
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
CUDA_VISIBLE_DEVICES=$GPU python train_actuator_diffsim.py \
    --train_config configs/no_load_with_gripper.yaml \
    --seed 0 \
    --log_dir outputs/logs_no_load_with_gripper \
    --model_out outputs/no_load_with_gripper_params.pkl
