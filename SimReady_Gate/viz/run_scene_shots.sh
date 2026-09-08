#!/bin/bash
cd "$(dirname "$0")/.."
${PYTHON:-python3} viz/make_robolab_scene_video.py --scene "${ROBOLAB_DIR:-../ext/RoboLab}/assets/scenes/workdesk_snacks.usda" --out results/videos/workdesk 2>&1 | grep --line-buffered -v '^\[\|Warning: in Bindings' | tee results/videos/workdesk_shots.log
