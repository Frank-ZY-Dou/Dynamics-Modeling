"""Check a custom pose without connecting to a hand."""

import argparse
from pathlib import Path

from wuji2_control.gestures import get_pose
from wuji2_control.model import JointModel
from wuji2_control.motion import Waypoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    args = parser.parse_args()
    start = get_pose("open_palm")
    target = start.copy()
    target[4] = 0.15
    waypoint = Waypoint("small_index_curl", target, hold_s=1.0)
    model = JointModel(args.model, handedness="left")
    model.preflight(start, waypoint.q)
    model.preflight(waypoint.q, start)
    print(waypoint.q.tolist())


if __name__ == "__main__":
    main()
