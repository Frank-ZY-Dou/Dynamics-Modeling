#!/bin/bash
# usage: bash scripts/train_so101_extended_ft.sh [gpu_id]
GPU=${1:-0}
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
CUDA_VISIBLE_DEVICES=$GPU python train_actuator_diffsim_so101.py \
    --train_config configs/so101_extended_ft.yaml \
    --log_dir outputs/logs_so101_extended_ft \
    --model_out outputs/so101_extended_ft_params.pkl
