#!/bin/bash
cd "$(dirname "$0")/.."
${PYTHON:-python3} viz/make_robolab_scene_video.py --layout 10 2 --out results/videos/layout_n10_s2 2>&1 | grep --line-buffered -v '^\[\|Warning: in Bindings\|Polishing' | tee results/videos/layout_n10_s2_shots.log
