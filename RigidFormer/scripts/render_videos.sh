#!/bin/bash
# Rollout videos of a checkpoint (the training code's video path in inference mode): test scenes only, the given step
# sizes (default 1), the checkpoint's own training config. OVERLAY=1 adds the ground truth as a translucent ghost;
# FLOOR=dark renders on a dark floor (default light).
#
#   CHECKPOINT=ckpt.pth BASE_CONFIG=configs/movi_sphere.yaml OUT=videos/S_ep100 [STEP_SIZES=1] [N_TEST=2] \
#       [OVERLAY=0] [FLOOR=light] [GPU=0] bash scripts/render_videos.sh
#
# The mp4 files are moved to OUT (a few MB each); the run's log stays in OUT/render.log.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${CHECKPOINT:?}" "${BASE_CONFIG:?}" "${OUT:?}"
STEP_SIZES="${STEP_SIZES:-1}"; N_TEST="${N_TEST:-2}"; GPU="${GPU:-0}"; PORT="${PORT:-29950}"; OVERLAY="${OVERLAY:-0}"; FLOOR="${FLOOR:-light}"
mkdir -p "$OUT"
TAG="render_$(date +%Y%m%d_%H%M%S)_$$"
CFG="configs/_tmp_${TAG}.yaml"
python - "$BASE_CONFIG" "$CFG" "$TAG" "$STEP_SIZES" "$N_TEST" "$OVERLAY" "$FLOOR" <<'EOF'
import sys, yaml
base, out, tag, step_sizes, n_test, overlay, floor = sys.argv[1:]
c = yaml.safe_load(open(base))
c['experiment_tag'] = tag
c['data']['eval_step_sizes'] = [int(s) for s in step_sizes.split(',')]
ck = c['training']['checkpoint']
ck['save_video'] = True
ck['if_overlay'] = overlay == '1'
ck['video_floor'] = floor
ck['num_videos_per_split'] = {'train': 0, 'val': 0, 'test': int(n_test)}
c.setdefault('inference', {}).update({'mode': True, 'generate_videos': True, 'compute_metrics': False})
c['training']['enable_validation'] = False
yaml.safe_dump(c, open(out, 'w'), sort_keys=False)
EOF
rc=0
CUDA_VISIBLE_DEVICES=$GPU PYOPENGL_PLATFORM=egl MPLBACKEND=Agg torchrun --standalone --nproc_per_node=1 --master_port=$PORT \
  src/train.py --config "$CFG" --checkpoint "$CHECKPOINT" --inference --use_multi_dataset > "$OUT/render.log" 2>&1 || rc=$?
LOGDIR=$(ls -td exps/logs_multi/$TAG/run_* 2>/dev/null | head -1)
if [ -n "$LOGDIR" ]; then find "$LOGDIR" -name '*.mp4' -exec mv {} "$OUT/" \; ; fi
rm -f "$CFG"
ls -la "$OUT"
# success only with every requested video present and non-empty (N_TEST scenes x step sizes)
want=$(( N_TEST * $(echo "$STEP_SIZES" | tr ',' '\n' | grep -c .) ))
have=$(find "$OUT" -maxdepth 1 -name '*.mp4' -size +0 | wc -l)
if [ $rc -eq 0 ] && [ "$have" -lt "$want" ]; then echo "render_videos: $have of $want videos written" >&2; rc=1; fi
exit $rc
