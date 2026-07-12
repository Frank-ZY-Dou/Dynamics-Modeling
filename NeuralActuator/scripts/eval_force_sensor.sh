#!/bin/bash
# usage: bash scripts/eval_force_sensor.sh [checkpoint] [gpu_id]
CKPT=${1:-outputs/force_sensor_params_best_val.pkl}
GPU=${2:-0}
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
CUDA_VISIBLE_DEVICES=$GPU python evaluate_actuator.py \
    --model_path "$CKPT" \
    --config configs/force_sensor.yaml \
    --output outputs/force_sensor_eval.json \
    --dump_rollout outputs/force_sensor_rollouts
