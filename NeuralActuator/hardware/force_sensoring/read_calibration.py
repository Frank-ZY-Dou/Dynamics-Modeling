#!/usr/bin/env python3
"""
Read calibration constants from BonAppetit force sensor
Attempts to read calibration matrix directly via serial port
"""

import serial
import json
import time

PORT = '/dev/ttyACM0'
BAUDRATE = 115200
TIMEOUT = 1.0

def send_command(ser, command, args=None):
    """Send JSON command to sensor and get response"""
    if args is None:
        args = {}

    cmd = {
        "command": command,
        "command_args": args
    }

    cmd_str = json.dumps(cmd) + '\n'
    print(f"\n-> Sending: {cmd_str.strip()}")

    # Clear input buffer
    ser.reset_input_buffer()

    # Send command
    ser.write(cmd_str.encode('utf-8'))
    ser.flush()

    # Wait for response
    time.sleep(0.2)

    # Read response
    responses = []
    start_time = time.time()
    while time.time() - start_time < TIMEOUT:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line:
                responses.append(line)
                print(f"← Response: {line}")
                # Try to parse as JSON
                try:
                    data = json.loads(line)
                    return data
                except:
                    pass

    if responses:
        return responses
    return None

def main():
    print(f"Connecting to {PORT} at {BAUDRATE} baud...")

    try:
        with serial.Serial(PORT, BAUDRATE, timeout=TIMEOUT) as ser:
            print("Connected!\n")
            time.sleep(0.5)

            # Try to get firmware version
            print("=" * 60)
            print("1. Getting firmware version...")
            send_command(ser, "get_firmware_version")

            # Try to get serial number
            print("\n" + "=" * 60)
            print("2. Getting serial number...")
            send_command(ser, "get_serial_number")

            # Try to get ADC sensor output model
            print("\n" + "=" * 60)
            print("3. Getting ADC sensor output model...")
            send_command(ser, "get_adc_sensor_output_model")

            # Try to get calibration constants (index 0-39 based on CalibrationConstant enum)
            print("\n" + "=" * 60)
            print("4. Trying to get calibration constants...")
            print("   (CalibrationConstant has indices 0-39)")

            # Try a few indices
            for idx in [0, 1, 2, 3, 4, 5]:
                print(f"\n   Calibration constant index {idx}:")
                send_command(ser, "get_calibration_constant", {"calibration_constant_index": idx})
                time.sleep(0.1)

            # Try to get all calibration constants
            print("\n" + "=" * 60)
            print("5. Trying to get all calibration constants at once...")
            send_command(ser, "get_all_calibration_constants")

            # Try to get data rate
            print("\n" + "=" * 60)
            print("6. Getting data rate...")
            send_command(ser, "get_data_rate")

            # Try to get device mode
            print("\n" + "=" * 60)
            print("7. Getting device mode...")
            send_command(ser, "get_mode")

            print("\n" + "=" * 60)
            print("Done!")

    except serial.SerialException as e:
        print(f"Serial error: {e}")
    except KeyboardInterrupt:
        print("\nInterrupted by user")

if __name__ == "__main__":
    main()
