#!/bin/bash
cd "$(dirname "$0")/.."
${PYTHON:-python3} viz/make_robocasa_video.py --n 12 --region 0.2 --seed 1 --out results/videos/robocasa_n12 2>&1 | grep --line-buffered -v '^\[' | tee results/videos/robocasa_n12.log
