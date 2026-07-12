#!/usr/bin/env python
"""
Direct serial reading for force sensor (bypass BonAppetit API).
For sensors with unsupported firmware versions.
"""

import serial
import time
import argparse


def read_sensor_direct(port="/dev/ttyACM0", baudrate=115200, duration=10):
    """
    Read force sensor data directly from serial port.

    Args:
        port: Serial port path
        baudrate: Communication baudrate
        duration: How long to read (seconds), None = infinite
    """
    print(f"Opening serial port {port} at {baudrate} baud...")

    try:
        ser = serial.Serial(
            port=port,
            baudrate=baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=1.0
        )

        print(f"Connected to {port}")
        print("Reading data... (Press Ctrl+C to stop)\n")

        start_time = time.time()
        line_count = 0

        while True:
            # Check duration
            if duration is not None and (time.time() - start_time) >= duration:
                break

            # Read line from serial
            if ser.in_waiting > 0:
                try:
                    # Try reading a line
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                    if line:
                        line_count += 1
                        print(f"[{line_count}] {line}")
                except UnicodeDecodeError:
                    # Try reading as bytes
                    raw_data = ser.read(ser.in_waiting)
                    print(f"[{line_count}] Raw bytes: {raw_data.hex()}")
            else:
                time.sleep(0.01)

        elapsed = time.time() - start_time
        print(f"\n\nRead {line_count} lines in {elapsed:.2f} seconds")
        print(f"Average rate: {line_count/elapsed:.1f} lines/sec")

    except serial.SerialException as e:
        print(f"Serial port error: {e}")
        print(f"\nTroubleshooting:")
        print(f"  1. Check port exists: ls /dev/ttyACM*")
        print(f"  2. Check permissions: sudo chmod 666 {port}")
        print(f"  3. Check if another program is using the port")
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()
            print("Serial port closed")


def send_command(port="/dev/ttyACM0", baudrate=115200, command=""):
    """
    Send a command to the sensor and read response.

    Common commands for force sensors:
    - Get firmware version
    - Start/stop streaming
    - Reset/tare
    """
    print(f"Opening serial port {port}...")

    try:
        ser = serial.Serial(port, baudrate, timeout=2.0)
        print(f"Connected. Sending command: {command}")

        # Send command
        ser.write((command + '\r\n').encode())
        time.sleep(0.2)

        # Read response
        response_lines = []
        while ser.in_waiting > 0:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line:
                response_lines.append(line)
                print(f"  Response: {line}")

        if not response_lines:
            print("  No response received")

        ser.close()
        return response_lines

    except serial.SerialException as e:
        print(f"Error: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Direct serial force sensor reader")
    parser.add_argument(
        "--port",
        default="/dev/ttyACM0",
        help="Serial port (default: /dev/ttyACM0)"
    )
    parser.add_argument(
        "--baudrate",
        type=int,
        default=115200,
        help="Baudrate (default: 115200)"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Read duration in seconds (default: 10)"
    )
    parser.add_argument(
        "--command",
        type=str,
        default=None,
        help="Send a command instead of reading (e.g., 'v' for version)"
    )
    parser.add_argument(
        "--infinite",
        action="store_true",
        help="Read indefinitely (ignore duration)"
    )

    args = parser.parse_args()

    if args.command:
        # Command mode
        send_command(args.port, args.baudrate, args.command)
    else:
        # Reading mode
        duration = None if args.infinite else args.duration
        read_sensor_direct(args.port, args.baudrate, duration)


if __name__ == "__main__":
    main()
