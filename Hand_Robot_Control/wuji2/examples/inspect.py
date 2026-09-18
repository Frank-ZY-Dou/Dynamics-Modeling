"""Read feedback without enabling motors or changing control settings."""

import argparse
import json
from dataclasses import asdict

from wuji2_control import ConnectionConfig, Wuji2Device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", required=True)
    args = parser.parse_args()
    device = Wuji2Device(ConnectionConfig(serial=args.serial, handedness="left"))
    try:
        device.connect()
        print(json.dumps(asdict(device.wait_for_snapshot()), indent=2))
    finally:
        device.close()


if __name__ == "__main__":
    main()
