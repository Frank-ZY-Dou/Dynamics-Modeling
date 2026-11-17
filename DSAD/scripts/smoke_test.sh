#!/usr/bin/env bash
# Smoke test: fit Bunny GoP 1 for 500 + 500 iterations with configs/dsad.yaml, export its stored
# form and decode it. Under a minute once setup.sh has built the extensions. The fit is far from
# converged: about 30 dB, against 38.7 dB for the full 60k fit.
#
#     CUDA_VISIBLE_DEVICES=0 bash scripts/smoke_test.sh
#
# GOP=<dir> sets the GoP, OUT=<dir> the output root (default checkpoints) and PYTHON the
# interpreter. An earlier smoke run of the same GoP is moved to <OUT>/smoke/stale/.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-$(command -v python || command -v python3 || true)}"
[ -n "$PY" ] || { echo "no python found; set PYTHON=/path/to/python" >&2; exit 1; }
GOP="${GOP:-baselines/D2GV/dataset/bunny/bunny_1}"
OUT="${OUT:-checkpoints}"
cd "$REPO"

RUN="$OUT/smoke/SAD3_500_27047/$(basename "$GOP")"
if [ -e "$RUN" ]; then
    STALE="$OUT/smoke/stale/$(date -u +%Y%m%dT%H%M%SZ)"
    mkdir -p "$STALE"
    mv "$RUN" "$STALE/"
fi

"$PY" dsad/train_sad_video.py --config configs/dsad.yaml -d "$GOP" \
    --num_points 27047 --iterations 500 --coarse_iterations 500 --cand_exact_tail 100 \
    --data_name smoke --output_dir "$OUT"

"$PY" dsad/export.py --run "$RUN"
"$PY" dsad/decode.py --run "$RUN" --warmup 2 --repeats 5 --json "$RUN/decode_smoke.json"
