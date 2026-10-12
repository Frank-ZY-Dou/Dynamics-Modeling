#!/bin/bash
# Evaluate a trained RigidFormer checkpoint.
# Runs cross-dataset evaluation on MOVi-B (in-domain) and optionally
# MOVi-A / MOVi-Sphere (cross-domain generalization).
#
# Usage:
#   ./scripts/eval.sh
#   CHECKPOINT=checkpoints/rigidformer_movib.pth TARGETS=B,A,S GPUS=0 ./scripts/eval.sh
#
# Environment overrides:
#   CHECKPOINT   Path to the model checkpoint (default: checkpoints/rigidformer_movib.pth)
#   BASE_CONFIG  Base training config to derive eval configs from (default: configs/movi_b.yaml)
#   TARGETS      Comma-separated dataset keys: B, A, S (default: B)
#   GPUS         CUDA_VISIBLE_DEVICES string, e.g. "0" or "0,1" (default: 0)
#   OUTPUT_FILE  Consolidated results file (default: eval_results.txt)
#   COMPILE      Set to 1 to compile the model with torch.compile (faster after
#                a one-time compilation; see README)
#   CUDA_GRAPH   Set to 1 to replay the rollout step as a CUDA graph (about twice
#                as fast; trajectories identical to the standard path)
#   SPLIT        test (default, reported numbers) or val: evaluate the validation
#                scenes with the same protocol, for checkpoint selection
#   POSITION_ESTIMATOR  config (default), area_centroid (area-weighted surface centroid
#                of the predicted mesh) or hopnet (HopNet's shape matching, exact for
#                a perfect prediction)
#   RENDER       After the metrics, render the predictions on this many test scenes
#                per step size (default 3; 0 = no videos)
#   FLOOR        light (default) or dark floor in the videos
#   VIDEO_DIR    Videos: VIDEO_DIR/<dataset>/scene_<n>_step_<s>.mp4 (default: videos)
set -euo pipefail

cd "$(dirname "$0")/.."

CHECKPOINT="${CHECKPOINT:-checkpoints/rigidformer_movib.pth}"
BASE_CONFIG="${BASE_CONFIG:-configs/movi_b.yaml}"
TARGETS="${TARGETS:-B}"
GPUS="${GPUS:-0}"
OUTPUT_FILE="${OUTPUT_FILE:-eval_results.txt}"
COMPILE="${COMPILE:-0}"
CUDA_GRAPH="${CUDA_GRAPH:-0}"
SPLIT="${SPLIT:-test}"
POSITION_ESTIMATOR="${POSITION_ESTIMATOR:-config}"
RENDER="${RENDER:-3}"
FLOOR="${FLOOR:-light}"
VIDEO_DIR="${VIDEO_DIR:-videos}"

if [ ! -f "$CHECKPOINT" ]; then
  echo "ERROR: checkpoint not found: $CHECKPOINT" >&2
  exit 1
fi
if [ ! -f "$BASE_CONFIG" ]; then
  echo "ERROR: base config not found: $BASE_CONFIG" >&2
  exit 1
fi

echo "[eval] checkpoint  = $CHECKPOINT"
echo "[eval] base_config = $BASE_CONFIG"
echo "[eval] targets     = $TARGETS"
echo "[eval] gpus        = $GPUS"
echo "[eval] output      = $OUTPUT_FILE"
echo "[eval] split       = $SPLIT"
echo "[eval] estimator   = $POSITION_ESTIMATOR"
echo "[eval] render      = $RENDER test scenes per step size ($FLOOR floor) -> $VIDEO_DIR"

python scripts/eval_cross_dataset.py \
  --base_config "$BASE_CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --output_file "$OUTPUT_FILE" \
  --targets "$TARGETS" \
  --gpus "$GPUS" \
  --split "$SPLIT" \
  --position_estimator "$POSITION_ESTIMATOR" \
  --render "$RENDER" \
  --floor "$FLOOR" \
  --video_dir "$VIDEO_DIR" \
  $([ "$COMPILE" = "1" ] && echo --compile) \
  $([ "$CUDA_GRAPH" = "1" ] && echo --cuda-graph)

echo "[eval] Done. Results written to $OUTPUT_FILE"
