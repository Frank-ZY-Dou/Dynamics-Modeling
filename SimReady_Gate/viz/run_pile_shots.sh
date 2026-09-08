#!/bin/bash
cd "$(dirname "$0")/.."
${PYTHON:-python3} viz/make_robolab_scene_video.py --pile 20 2 --program results/videos/pile_n20/program.json --orbit 40 --s-min 0.3 --repair-seconds 8 --hold-seconds 2.5 --settle-seconds 3.0 --out results/videos/pile_n20 2>&1 | grep --line-buffered -v '^\[\|Warning: in Bindings\|Polishing' | tee results/videos/pile_n20_shots.log
