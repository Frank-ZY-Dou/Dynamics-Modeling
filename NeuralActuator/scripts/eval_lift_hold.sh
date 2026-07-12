#!/bin/bash
# usage: bash scripts/eval_lift_hold.sh [checkpoint] [gpu_id]
CKPT=${1:-outputs/lift_hold_params_best_val.pkl}
GPU=${2:-0}
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
CUDA_VISIBLE_DEVICES=$GPU python evaluate_actuator.py \
    --model_path "$CKPT" \
    --config configs/lift_hold.yaml \
    --output outputs/lift_hold_eval.json \
    --dump_rollout outputs/lift_hold_rollouts
