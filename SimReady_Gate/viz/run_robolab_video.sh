#!/bin/bash
cd "$(dirname "$0")/.."
${PYTHON:-python3} viz/make_robolab_video.py --n 10 --seed 0 --out results/videos/n10_s0 2>&1 | grep --line-buffered -v '^\[' | tee results/videos/n10_s0.log
