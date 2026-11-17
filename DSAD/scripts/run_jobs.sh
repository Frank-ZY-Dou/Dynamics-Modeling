#!/usr/bin/env bash
# Run a job file (the output of scripts/make_jobs.py) on several GPUs. One worker per GPU takes the
# next line of the queue when its previous one ends.
#
#     PY=/path/to/python bash scripts/run_jobs.sh jobs/main.txt 0 1 2 3
#
# Lines run from the DSAD directory with CUDA_VISIBLE_DEVICES set to the worker's GPU, and training
# output goes to logs/<tag>.log there. The queue is <jobs>.queue/ next to the job file: pending.txt
# (lines not yet started), done.log (exit code and duration of every line that ended) and one log
# per worker. Run one training job per GPU: the fused step keeps a GPU busy.
#
# Ctrl-C stops the workers and their fits. Resuming: run the same command again. The queue is seeded from the job file, finished runs are
# skipped at once, and runs that were interrupted or ended with an error start over (the files of
# the unfinished attempt are moved into <run>/stale/). A run_jobs.sh started on the same job file
# while another one runs joins its queue: more GPUs on the same lines, and no line runs twice.
#
# Before the workers start, the CUDA extensions that the job lines use are loaded once on the first
# GPU listed, which builds any that setup.sh has not built. Workers must not build them at the same
# time: gsplat's loader deletes a build lock and an unfinished build directory that it finds. If a
# build was interrupted and jobs hang at "Loading extension", delete the stale `lock` file under
# .torch_extensions/.
#
# PY (or PYTHON) names the interpreter. The default is python, else python3, on PATH.
set -uo pipefail

[ $# -ge 2 ] || { echo "usage: $0 <jobs.txt> <gpu> [<gpu> ...]" >&2; exit 1; }
JOBS=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
shift
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PY="${PY:-${PYTHON:-$(command -v python || command -v python3 || true)}}"
[ -n "$PY" ] || { echo "no python found; set PY=/path/to/python" >&2; exit 1; }
[ -f "$JOBS" ] || { echo "no such job file: $JOBS" >&2; exit 1; }
"$PY" -c "import torch" 2>/dev/null || {
    echo "$PY cannot import torch: set PY to the Python of the environment DSAD is installed in" >&2; exit 1; }

QDIR="${JOBS%.txt}.queue"
mkdir -p "$QDIR" "$REPO/logs"
# Seed the queue from the job file, or join a run_jobs.sh that works on it. Every process of an
# invocation (its workers and their jobs included) holds a shared lock on .active, so a queue counts
# as running as long as any of its jobs does.
exec 9>>"$QDIR/.active"
if flock -n -x 9; then
    cp "$JOBS" "$QDIR/pending.txt"
    flock -s 9
    echo "queue seeded from $JOBS (finished runs are skipped)"
else
    flock -s 9
    echo "joining the queue of the run_jobs.sh that is running $JOBS"
fi
# Same JIT-extension cache as the Python entry points (dsad/_paths.py).
if [ -z "${TORCH_EXTENSIONS_DIR:-}" ]; then
    TORCH_EXTENSIONS_DIR=$("$PY" -c "import sys; sys.path.insert(0, sys.argv[1]); import _paths; print(_paths.extensions_dir())" "$REPO/dsad")
fi
export TORCH_EXTENSIONS_DIR

# Load (and if needed build) the extensions the pending lines use, once, before any worker starts:
# DSAD's fused step and exact candidate builder, and D2GV's gsplat rasterizer (DSAD-GS lines).
lines=$(grep -v '^\s*\(#\|$\)' "$QDIR/pending.txt" || true)
NEED_DSAD=0; NEED_GSPLAT=0
grep -q -E 'dsad/train_sad_video\.py' <<<"$lines" && NEED_DSAD=1
grep -q -E -e '--appearance gaussian|configs/dsad_gs\.yaml' <<<"$lines" && NEED_GSPLAT=1
if [ "$NEED_DSAD$NEED_GSPLAT" != 00 ]; then
    echo "loading the CUDA extensions of these jobs on GPU $1 (a missing build is compiled now) ..."
    (cd "$REPO" && CUDA_VISIBLE_DEVICES=$1 NEED_DSAD=$NEED_DSAD NEED_GSPLAT=$NEED_GSPLAT "$PY" - <<'PY'
import os, sys
sys.path.insert(0, "dsad")
import _paths
_paths.setup()
if os.environ["NEED_DSAD"] == "1":
    __import__("fast_step").ext()
    __import__("exact_candidates")._ext()
    print("  dsad_fast, dsad_exact_candidates: ok")
if os.environ["NEED_GSPLAT"] == "1":
    if __import__("gsplat.cuda._backend", fromlist=["_C"])._C is None:
        raise SystemExit("gsplat's CUDA extension did not load (is nvcc on PATH?)")
    print("  gsplat_cuda: ok")
PY
    ) || { echo "the CUDA extensions of these jobs do not load (see setup.sh)" >&2; exit 1; }
fi

pop() {  # print the next queue line and remove it; fail when the queue is empty
    flock "$QDIR/.lock" bash -c '[ -s "$0" ] || exit 1; head -n 1 "$0"; sed -i 1d "$0"' \
        "$QDIR/pending.txt"
}

worker() {
    local gpu=$1 n=0 job start rc
    while job=$(pop); do
        case "$job" in ''|'#'*) continue ;; esac
        n=$((n + 1))
        start=$(date +%s)
        echo "[gpu$gpu] START $(date +%T) :: ${job:0:160}"
        (cd "$REPO" && CUDA_VISIBLE_DEVICES=$gpu bash -c "$job")
        rc=$?
        flock "$QDIR/.lock" printf '%s gpu%s rc=%d %ds :: %s\n' "$(date +%FT%T)" "$gpu" "$rc" \
            "$(( $(date +%s) - start ))" "$job" >> "$QDIR/done.log"
        [ "$rc" -eq 0 ] || flock "$QDIR/.lock" printf '%s\n' "$job" >> "$FAILED"
        echo "[gpu$gpu] END   $(date +%T) rc=$rc"
    done
    echo "[gpu$gpu] queue empty after $n jobs"
}

echo "jobs: $JOBS ($(grep -cv '^\s*\(#\|$\)' "$QDIR/pending.txt") lines pending)  GPUs: $*  python: $PY"
FAILED="$QDIR/failed.$$"
kill_tree() { local c; for c in $(pgrep -P "$1"); do kill_tree "$c"; done; kill "$1" 2>/dev/null; }
k=0; pids=()
for gpu in "$@"; do     # a GPU listed twice gets two workers
    k=$((k + 1))
    worker "$gpu" >> "$QDIR/worker${k}_gpu$gpu.log" 2>&1 &
    pids+=($!)
done
trap 'for p in "${pids[@]}"; do kill_tree "$p"; done; echo "stopped: run the same command to resume" >&2; exit 130' INT TERM
wait
if [ -s "$FAILED" ]; then
    echo "$(wc -l < "$FAILED") fits failed (see logs/<tag>.log); run the same command to retry them:" >&2
    sed -n -E 's/.*--data_name ([^ ]+).*/  \1/p' "$FAILED" | head -20 >&2
    rm -f "$FAILED"
    exit 1
fi
echo "all workers finished; see $QDIR/done.log"
