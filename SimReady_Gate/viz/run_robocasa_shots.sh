#!/bin/bash
cd "$(dirname "$0")/.."
${PYTHON:-python3} viz/make_robocasa_video.py --n 10 --region 0.15 --seed 0 --out results/videos/robocasa_n10 2>&1 | grep --line-buffered -v '^\[\|Polishing' | tee results/videos/robocasa_n10_shots.log
