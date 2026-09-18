"""Run a recorded gesture through the shared controller.

Arguments and defaults are identical to ``wuji2 run``. Supply --execute to move.
"""

import sys

from wuji2_control.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["run", *sys.argv[1:]]))
