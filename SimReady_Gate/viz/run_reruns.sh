#!/bin/bash
cd "$(dirname "$0")/.."
P=${PYTHON:-python3}
case "$1" in
  diag)     $P experiments/robolab_diagnostic.py 4,6,8,10 0,1,2 2>&1 | grep --line-buffered -v '^\[\|Polishing\|Warning: in Bindings' | tee results/rerun_diag.log ;;
  g5)       $P experiments/robolab_g5_sweep.py 2>&1 | grep --line-buffered -v '^\[\|Polishing\|Warning: in Bindings' | tee results/rerun_g5.log ;;
  capacity) $P experiments/robocasa_capacity.py 2>&1 | grep --line-buffered -v '^\[\|Polishing' | tee results/rerun_capacity.log ;;
  shipped)  $P experiments/robolab_shipped_sweep.py 2>&1 | grep --line-buffered -v '^\[\|Warning: in Bindings' | tee results/rerun_shipped.log ;;
esac
if [ "$1" = "g0" ]; then $P experiments/robocasa_g0_sweep.py 2>&1 | grep --line-buffered -v '^\[' | tee results/rerun_g0.log; fi
