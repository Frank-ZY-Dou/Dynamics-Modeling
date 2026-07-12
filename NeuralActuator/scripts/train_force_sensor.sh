#!/bin/bash
# usage: bash scripts/train_force_sensor.sh [gpu_id]
GPU=${1:-0}
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
CUDA_VISIBLE_DEVICES=$GPU python train_actuator_diffsim.py \
    --train_config configs/force_sensor.yaml \
    --seed 0 \
    --log_dir outputs/logs_force_sensor \
    --model_out outputs/force_sensor_params.pkl
