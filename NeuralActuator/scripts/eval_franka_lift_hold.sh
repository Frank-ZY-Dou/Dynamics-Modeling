#!/bin/bash
# usage: bash scripts/eval_franka_lift_hold.sh [checkpoint] [gpu_id]
CKPT=${1:-outputs/franka_lift_hold_ft2_params_best_val.pkl}
GPU=${2:-0}
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
CUDA_VISIBLE_DEVICES=$GPU python evaluate_actuator_franka.py \
    --model_path "$CKPT" \
    --config configs/franka_lift_hold.yaml \
    --output outputs/franka_lift_hold_eval.json \
    --dump_rollout outputs/franka_lift_hold_rollouts
