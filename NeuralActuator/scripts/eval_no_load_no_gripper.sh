#!/bin/bash
# usage: bash scripts/eval_no_load_no_gripper.sh [checkpoint] [gpu_id]
CKPT=${1:-outputs/no_load_no_gripper_params_best_val.pkl}
GPU=${2:-0}
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
CUDA_VISIBLE_DEVICES=$GPU python evaluate_actuator.py \
    --model_path "$CKPT" \
    --config configs/no_load_no_gripper.yaml \
    --output outputs/no_load_no_gripper_eval.json \
    --dump_rollout outputs/no_load_no_gripper_rollouts
