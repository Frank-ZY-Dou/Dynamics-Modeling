#!/bin/bash
# usage: bash scripts/train_weight_all_ft1.sh [gpu_id]
GPU=${1:-0}
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
CUDA_VISIBLE_DEVICES=$GPU python train_actuator_diffsim.py \
    --train_config configs/weight_all_ft1.yaml \
    --log_dir outputs/logs_weight_all_ft1 \
    --model_out outputs/weight_all_ft1_params.pkl
