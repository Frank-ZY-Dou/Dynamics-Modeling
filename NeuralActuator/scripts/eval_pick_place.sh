#!/bin/bash
# usage: bash scripts/eval_pick_place.sh [checkpoint] [gpu_id]
CKPT=${1:-outputs/pick_place_params_best_val.pkl}
GPU=${2:-0}
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
CUDA_VISIBLE_DEVICES=$GPU python evaluate_actuator.py \
    --model_path "$CKPT" \
    --config configs/pick_place.yaml \
    --output outputs/pick_place_eval.json \
    --dump_rollout outputs/pick_place_rollouts
