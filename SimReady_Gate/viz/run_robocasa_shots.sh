#!/bin/bash
cd "$(dirname "$0")/.."
${PYTHON:-python3} viz/make_robocasa_video.py --n 10 --ring 0.2 --seed 0 --out results/videos/robocasa_ring 2>&1 | grep --line-buffered -v '^\[\|Polishing' | tee results/videos/robocasa_ring_shots.log
