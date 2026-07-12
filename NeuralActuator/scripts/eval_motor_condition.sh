#!/bin/bash
# usage: bash scripts/eval_motor_condition.sh [checkpoint] [gpu_id]
CKPT=${1:-outputs/motor_condition_params_best_val.pkl}
GPU=${2:-0}
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
CUDA_VISIBLE_DEVICES=$GPU python evaluate_actuator.py \
    --model_path "$CKPT" \
    --config configs/motor_condition.yaml \
    --output outputs/motor_condition_eval.json \
    --dump_rollout outputs/motor_condition_rollouts
