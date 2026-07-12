#!/bin/bash
# usage: bash scripts/eval_so101_weight.sh [checkpoint] [gpu_id]
CKPT=${1:-outputs/so101_weight_params_best_val.pkl}
GPU=${2:-0}
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
CUDA_VISIBLE_DEVICES=$GPU python evaluate_actuator_so101.py \
    --model_path "$CKPT" \
    --config configs/so101_weight.yaml \
    --output outputs/so101_weight_eval.json \
    --dump_rollout outputs/so101_weight_rollouts
