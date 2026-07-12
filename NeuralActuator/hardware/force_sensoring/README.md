# Force Sensor Reading

## Quick Start (Recommended)

Python dependencies: `pyserial`, plus `matplotlib` for `--vis`. Both are hardware-side
extras and are not in the top-level `requirements.txt`; install them with pip.

### Reading and visualization
```bash
conda activate neural_robot
cd hardware/force_sensoring
python force_read_parsed.py --vis 
```

This will:
- Connect to sensor at `/dev/ttyACM0` (115200 baud)
- Parse 6-axis force/torque data: Fx, Fy, Fz, Mx, My, Mz
- Display real-time data with statistics every second
- Press Ctrl+C to stop

### With 3D Visualization
```bash
python force_read_parsed.py --vis
```

This will:
- **Everything from text mode PLUS**:
- Display two 3D coordinate systems:
  - **Left**: Force vector (Fx, Fy, Fz) as red arrow
  - **Right**: Torque vector (Mx, My, Mz) as purple arrow
- Show current values and magnitudes in 3D
- X-axis (red), Y-axis (green), Z-axis (blue)
- Close the plot window to exit

## Options

```bash
# Enable 3D visualization
python force_read_parsed.py --vis

# Read for 10 seconds only (text mode)
python force_read_parsed.py --duration 10

# Read single sample and exit
python force_read_parsed.py --single

# Disable statistics display
python force_read_parsed.py --no-stats

# 3D visualization without statistics
python force_read_parsed.py --vis --no-stats

# Use different port/baudrate
python force_read_parsed.py --port /dev/ttyACM1 --baudrate 9600
```

## Hardware

- Default sensor port: `/dev/ttyACM0`
- Default baudrate: 115200
- Check port: `ls /dev/ttyACM*`
- Set permissions if needed: `sudo chmod 666 /dev/ttyACM0`

## Other Files

- `force_read_serial.py` - Raw serial reading without parsing (for debugging)
- `force_read_plot.py` - **DEPRECATED** (use `force_read_parsed.py --vis` instead)
- `read_calibration.py` - Attempt to read calibration data (firmware not supported)

**Note**: The vendor API does not support this sensor firmware (u1.6.1), so all
readers here parse the serial stream directly.
