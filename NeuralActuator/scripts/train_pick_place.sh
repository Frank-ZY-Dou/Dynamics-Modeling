#!/bin/bash
# usage: bash scripts/train_pick_place.sh [gpu_id]
GPU=${1:-0}
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
CUDA_VISIBLE_DEVICES=$GPU python train_actuator_diffsim.py \
    --train_config configs/pick_place.yaml \
    --seed 0 \
    --log_dir outputs/logs_pick_place \
    --model_out outputs/pick_place_params.pkl
