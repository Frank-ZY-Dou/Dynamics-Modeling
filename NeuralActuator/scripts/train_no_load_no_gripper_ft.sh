#!/bin/bash
# usage: bash scripts/train_no_load_no_gripper_ft.sh [gpu_id]
GPU=${1:-0}
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
CUDA_VISIBLE_DEVICES=$GPU python train_actuator_diffsim.py \
    --train_config configs/no_load_no_gripper_ft.yaml \
    --log_dir outputs/logs_no_load_no_gripper_ft \
    --model_out outputs/no_load_no_gripper_ft_params.pkl
