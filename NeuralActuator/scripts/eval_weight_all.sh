#!/bin/bash
# usage: bash scripts/eval_weight_all.sh [checkpoint] [gpu_id]
CKPT=${1:-outputs/weight_all_params_best_val.pkl}
GPU=${2:-0}
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
CUDA_VISIBLE_DEVICES=$GPU python evaluate_actuator.py \
    --model_path "$CKPT" \
    --config configs/weight_all.yaml \
    --output outputs/weight_all_eval.json \
    --dump_rollout outputs/weight_all_rollouts
