#!/bin/bash
# usage: run_split_render.sh <out_root>   - high-quality render of the gate frames split over 3 GPUs, then the settle panels
cd "$(dirname "$0")/.."
ROOT="$1"; SH="$ROOT/shots.json"
N=$(${PYTHON:-python3} -c "import json; print(len(json.load(open('$SH'))['gate']))")
C=$(( (N + 2) / 3 ))
rm -rf "$ROOT/blender_gate" "$ROOT/blender_settle"; mkdir -p "$ROOT/blender_gate"
for g in 0 1 2; do
  S=$(( g * C )); E=$(( (g + 1) * C )); [ $E -gt $N ] && E=$N
  tmux new -d -s "bl-split-$g" "BLENDER=${BLENDER:-blender} bash viz/run_blender_hq.sh $SH gate -1 $g $ROOT/blender_gate --start $S --end $E"
done
tmux new -d -s "bl-split-settle0" "while tmux ls | grep -q 'bl-split-1:'; do sleep 20; done; BLENDER=${BLENDER:-blender} bash viz/run_blender_hq.sh $SH settle 0 1 $ROOT/blender_settle"
tmux new -d -s "bl-split-settle1" "while tmux ls | grep -q 'bl-split-2:'; do sleep 20; done; BLENDER=${BLENDER:-blender} bash viz/run_blender_hq.sh $SH settle 1 2 $ROOT/blender_settle"
echo "launched gate frames $N in chunks of $C on 3 GPUs; settle panels queued on GPUs 1 and 2"
