#!/bin/bash
# Train RigidFormer on MOVi-B with multi-GPU DDP (torchrun).
#
# Usage:
#   ./scripts/train.sh                          # train from scratch on all visible GPUs
#   GPUS=0,1,2,3 ./scripts/train.sh             # select specific GPUs
#   NPROC=4 ./scripts/train.sh                  # override the number of DDP processes
#   ./scripts/train.sh --resume                 # resume from the newest exps/checkpoints_multi/<tag>/exp_*/latest.pth
#   ./scripts/train.sh --checkpoint=ckpt.pth    # resume from a specific checkpoint
#   LR=1e-4 ./scripts/train.sh                  # override the learning rate
set -euo pipefail

cd "$(dirname "$0")/.."

# default: MOVi-B (configs/movi_sphere.yaml, configs/movi_a.yaml for the other datasets)
CONFIG="${CONFIG:-configs/movi_b.yaml}"

# GPU selection: GPUS sets CUDA_VISIBLE_DEVICES; default is all visible GPUs.
if [ -n "${GPUS:-}" ]; then
  export CUDA_VISIBLE_DEVICES="$GPUS"
fi
if [ -z "${NPROC:-}" ]; then
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    NPROC=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
  else
    NPROC=$(nvidia-smi -L 2>/dev/null | wc -l)
    if [ "$NPROC" -lt 1 ]; then NPROC=1; fi
  fi
fi

# Headless rendering backends for rollout videos.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

# Optional flags: --resume, --checkpoint=<path>, --lr=<value>
RESUME=false
CHECKPOINT_PATH=""
for arg in "$@"; do
  case $arg in
    --resume) RESUME=true ;;
    --checkpoint=*) CHECKPOINT_PATH="${arg#*=}"; RESUME=true ;;
    --lr=*) export LR="${arg#*=}" ;;
    *) echo "WARNING: ignoring unknown argument: $arg" >&2 ;;
  esac
done

CHECKPOINT_ARG=""
if [ "$RESUME" = true ]; then
  if [ -n "$CHECKPOINT_PATH" ]; then
    if [ ! -f "$CHECKPOINT_PATH" ]; then
      echo "ERROR: checkpoint not found: $CHECKPOINT_PATH" >&2
      exit 1
    fi
    CHECKPOINT_ARG="--checkpoint $CHECKPOINT_PATH"
  else
    # the newest run of this config's experiment tag: exps/checkpoints_multi/<experiment_tag>/exp_<timestamp>/latest.pth
    TAG=$(python -c "import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))['experiment_tag'])" "$CONFIG")
    LATEST=$(ls -td ./exps/checkpoints_multi/"$TAG"/exp_*/ 2>/dev/null | while read -r d; do
      [ -f "${d}latest.pth" ] && { echo "${d}latest.pth"; break; }; done)
    if [ -z "$LATEST" ]; then
      echo "ERROR: --resume given but no exps/checkpoints_multi/$TAG/exp_*/latest.pth found" >&2
      exit 1
    fi
    CHECKPOINT_ARG="--checkpoint $LATEST"
  fi
  echo "[train] Resuming from checkpoint: ${CHECKPOINT_ARG#--checkpoint }"
else
  echo "[train] Fresh training (no checkpoint)"
fi

mkdir -p ./exps/logs_multi ./exps/checkpoints_multi

echo "[train] config = $CONFIG"
echo "[train] gpus   = ${CUDA_VISIBLE_DEVICES:-all visible} (nproc_per_node=$NPROC)"
if [ -n "${LR:-}" ]; then
  echo "[train] learning rate override: LR=$LR"
fi

PYTHONUNBUFFERED=1 torchrun --standalone --nproc_per_node="$NPROC" \
  src/train.py \
  --config "$CONFIG" \
  $CHECKPOINT_ARG \
  --use_multi_dataset
