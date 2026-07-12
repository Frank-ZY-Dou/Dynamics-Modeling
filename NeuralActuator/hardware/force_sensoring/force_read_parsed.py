#!/usr/bin/env python
"""
Parse force sensor data from serial port.
Extracts 6-axis force/torque data.
Supports both text output and 3D visualization.
"""

import serial
import time
import argparse
import re
import os
import numpy as np
import csv
from datetime import datetime

# Fixed output directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "force_dataset")

# 3D visualization imports (optional)
try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.animation import FuncAnimation
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class ForceSensorReader:
    """Reader for 6-axis force/torque sensor."""

    def __init__(self, port="/dev/ttyACM0", baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        # Tare offset (initialized to zero)
        self.offset = {'fx': 0, 'fy': 0, 'fz': 0, 'mx': 0, 'my': 0, 'mz': 0}
        # Data saving
        self.output_file = None
        self.csv_file = None
        self.csv_writer = None
        self.start_time = None

    def connect(self):
        """Connect to sensor."""
        print(f"Connecting to {self.port} at {self.baudrate} baud...")
        self.ser = serial.Serial(
            port=self.port,
            baudrate=self.baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=1.0
        )
        print("Connected!")

    def disconnect(self):
        """Disconnect from sensor."""
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("Disconnected")

    def start_recording(self):
        """Start recording data to CSV file."""
        if self.output_file is None:
            return
        self.csv_file = open(self.output_file, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        # Write header
        self.csv_writer.writerow(['timestamp', 'fx', 'fy', 'fz', 'mx', 'my', 'mz'])
        self.start_time = time.time()
        print(f"[Recording] Saving data to: {self.output_file}")

    def stop_recording(self):
        """Stop recording and close file."""
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None
            print(f"[Recording] Data saved to: {self.output_file}")

    def save_data(self, data):
        """Save a single data point to CSV."""
        if self.csv_writer and data:
            timestamp = time.time() - self.start_time
            self.csv_writer.writerow([
                f"{timestamp:.4f}",
                f"{data['fx']:.4f}",
                f"{data['fy']:.4f}",
                f"{data['fz']:.4f}",
                f"{data['mx']:.4f}",
                f"{data['my']:.4f}",
                f"{data['mz']:.4f}"
            ])

    def parse_line(self, line):
        """
        Parse sensor data line.
        Format: < val1 val2 val3 val4 val5 val6 >

        Returns:
            dict with keys: fx, fy, fz, mx, my, mz
            or None if parsing failed
        """
        # Look for pattern < num num num num num num >
        pattern = r'<\s*(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)\s*>'
        match = re.search(pattern, line)

        if match:
            values = [int(match.group(i)) for i in range(1, 7)]
            return {
                'fx': values[0],
                'fy': values[1],
                'fz': values[2],
                'mx': values[3],
                'my': values[4],
                'mz': values[5]
            }
        return None

    def apply_offset(self, data):
        """
        Apply tare offset and convert to SI units.

        Raw units: mN (millinewtons), Nmm (newton-millimeters)
        Output units: N (newtons), Nm (newton-meters)
        """
        if data is None:
            return None
        return {
            # Force: mN -> N (divide by 1000)
            'fx': (data['fx'] - self.offset['fx']) / 1000.0,
            'fy': (data['fy'] - self.offset['fy']) / 1000.0,
            'fz': (data['fz'] - self.offset['fz']) / 1000.0,
            # Torque: Nmm -> Nm (divide by 1000)
            'mx': (data['mx'] - self.offset['mx']) / 1000.0,
            'my': (data['my'] - self.offset['my']) / 1000.0,
            'mz': (data['mz'] - self.offset['mz']) / 1000.0,
        }

    def tare(self, duration=10.0):
        """
        Perform tare (zero) operation.
        Collects data for specified duration and sets average as offset.

        Args:
            duration: How long to collect data (seconds), default 3s
        """
        if not self.ser or not self.ser.is_open:
            raise RuntimeError("Not connected")

        print(f"\n[Tare] Collecting data for {duration:.1f} seconds...")
        print("[Tare] Keep the sensor unloaded!\n")

        tare_buffer = {k: [] for k in ['fx', 'fy', 'fz', 'mx', 'my', 'mz']}
        start_time = time.time()

        while (time.time() - start_time) < duration:
            if self.ser.in_waiting > 0:
                line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                data = self.parse_line(line)
                if data:
                    for key in tare_buffer:
                        tare_buffer[key].append(data[key])
                    # Progress indicator
                    elapsed = time.time() - start_time
                    print(f"\r[Tare] {elapsed:.1f}s / {duration:.1f}s - Samples: {len(tare_buffer['fx'])}", end='')
            else:
                time.sleep(0.001)

        print()  # New line after progress

        if not tare_buffer['fx']:
            print("[Tare] ERROR: No data collected!")
            return False

        # Calculate average as offset
        for key in self.offset:
            self.offset[key] = int(np.mean(tare_buffer[key]))

        print(f"\n[Tare] Offset set:")
        print(f"  Force:  Fx={self.offset['fx']:6d}  Fy={self.offset['fy']:6d}  Fz={self.offset['fz']:6d}")
        print(f"  Torque: Mx={self.offset['mx']:6d}  My={self.offset['my']:6d}  Mz={self.offset['mz']:6d}")
        print(f"[Tare] Done! Collected {len(tare_buffer['fx'])} samples.\n")
        return True

    def read_single(self):
        """Read a single data packet."""
        if not self.ser or not self.ser.is_open:
            raise RuntimeError("Not connected")

        while True:
            if self.ser.in_waiting > 0:
                line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                data = self.parse_line(line)
                if data:
                    return data
            time.sleep(0.001)

    def read_continuous(self, duration=None, callback=None, show_stats=True):
        """
        Read sensor data continuously.

        Args:
            duration: How long to read (seconds), None = infinite
            callback: Function to call with each data dict
            show_stats: Show statistics every second
        """
        if not self.ser or not self.ser.is_open:
            raise RuntimeError("Not connected")

        print("Reading data... (Press Ctrl+C to stop)\n")

        start_time = time.time()
        last_stats_time = start_time
        count = 0

        # For statistics
        data_buffer = {k: [] for k in ['fx', 'fy', 'fz', 'mx', 'my', 'mz']}

        try:
            while True:
                # Check duration
                if duration and (time.time() - start_time) >= duration:
                    break

                # Read data
                if self.ser.in_waiting > 0:
                    line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                    data = self.parse_line(line)

                    if data:
                        # Apply tare offset
                        data = self.apply_offset(data)
                        count += 1

                        # Save to file if recording
                        self.save_data(data)

                        # Store for statistics
                        for key in data_buffer:
                            data_buffer[key].append(data[key])

                        # Call callback if provided
                        if callback:
                            callback(data)
                        else:
                            # Default: print data (SI units: N, Nm)
                            print(f"[{count:4d}] Fx:{data['fx']:7.3f}N Fy:{data['fy']:7.3f}N Fz:{data['fz']:7.3f}N "
                                  f"Mx:{data['mx']:6.3f}Nm My:{data['my']:6.3f}Nm Mz:{data['mz']:6.3f}Nm")

                        # Show statistics every second
                        if show_stats and (time.time() - last_stats_time) >= 1.0:
                            self._print_stats(data_buffer)
                            # Clear buffer
                            data_buffer = {k: [] for k in ['fx', 'fy', 'fz', 'mx', 'my', 'mz']}
                            last_stats_time = time.time()
                else:
                    time.sleep(0.001)

        except KeyboardInterrupt:
            print("\n\nStopped by user")

        elapsed = time.time() - start_time
        print(f"\nRead {count} samples in {elapsed:.2f} seconds")
        print(f"Average rate: {count/elapsed:.1f} Hz")

    def _print_stats(self, data_buffer):
        """Print statistics for the last second (SI units: N, Nm)."""
        if not data_buffer['fx']:
            return

        print("\n--- Stats (last 1s) ---")
        for key in ['fx', 'fy', 'fz']:
            values = data_buffer[key]
            unit = "N"
            print(f"{key.upper()}: mean={np.mean(values):7.3f}{unit} std={np.std(values):6.3f} "
                  f"min={np.min(values):7.3f} max={np.max(values):7.3f}")
        for key in ['mx', 'my', 'mz']:
            values = data_buffer[key]
            unit = "Nm"
            print(f"{key.upper()}: mean={np.mean(values):7.3f}{unit} std={np.std(values):6.3f} "
                  f"min={np.min(values):7.3f} max={np.max(values):7.3f}")
        print()


class ForceSensor3DVisualizer:
    """3D visualization wrapper for ForceSensorReader."""

    # Fixed axis limits
    FORCE_LIMIT = 50    # N
    TORQUE_LIMIT = 0.5  # Nm

    def __init__(self, reader):
        self.reader = reader
        self.fig = None
        self.ax_force = None
        self.ax_torque = None
        self.force_text = None
        self.torque_text = None

    def init_plot(self):
        """Initialize the 3D plot."""
        self.fig = plt.figure(figsize=(14, 7))

        # Create two subplots: Force (left) and Torque (right)
        self.ax_force = self.fig.add_subplot(121, projection='3d')
        self.ax_torque = self.fig.add_subplot(122, projection='3d')

        # Setup Force plot (larger range for N)
        self._setup_axis(self.ax_force, "Force (N)", self.FORCE_LIMIT)

        # Setup Torque plot (smaller range for Nm)
        self._setup_axis(self.ax_torque, "Torque (Nm)", self.TORQUE_LIMIT)

        # Text for values
        self.force_text = self.fig.text(0.02, 0.98, '', transform=self.fig.transFigure,
                                       verticalalignment='top', fontfamily='monospace',
                                       fontsize=10,
                                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        self.torque_text = self.fig.text(0.52, 0.98, '', transform=self.fig.transFigure,
                                        verticalalignment='top', fontfamily='monospace',
                                        fontsize=10,
                                        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

        plt.tight_layout()

    def _setup_axis(self, ax, title, limit):
        """Setup a 3D axis with specified limit."""
        # Set labels
        ax.set_xlabel('X', fontsize=12, fontweight='bold', color='red')
        ax.set_ylabel('Y', fontsize=12, fontweight='bold', color='green')
        ax.set_zlabel('Z', fontsize=12, fontweight='bold', color='blue')
        ax.set_title(title, fontsize=14, fontweight='bold')

        # Set axis limits (Y axis inverted for left-hand coordinate system)
        ax.set_xlim([-limit, limit])
        ax.set_ylim([limit, -limit])  # Inverted Y axis
        ax.set_zlim([-limit, limit])

        # Make axis lines thicker
        ax.xaxis.line.set_linewidth(4)
        ax.yaxis.line.set_linewidth(4)
        ax.zaxis.line.set_linewidth(4)

        # Draw coordinate axes reference lines (thicker)
        # Left-hand coordinate system with inverted Y axis
        ax.plot([0, limit*0.8], [0, 0], [0, 0], 'r-', alpha=0.5, linewidth=6)
        ax.plot([0, 0], [0, limit*0.8], [0, 0], 'g-', alpha=0.5, linewidth=6)
        ax.plot([0, 0], [0, 0], [0, limit*0.8], 'b-', alpha=0.5, linewidth=6)

        # Grid
        ax.grid(True, alpha=0.2)

    def update_plot(self, frame, data_callback):
        """Update function for animation."""
        # Get latest data from reader
        data = data_callback()

        if data is None:
            return []

        fx, fy, fz = data['fx'], data['fy'], data['fz']
        mx, my, mz = data['mx'], data['my'], data['mz']

        # Update Force plot
        self.ax_force.cla()
        self._setup_axis(self.ax_force, "Force (N)", self.FORCE_LIMIT)

        if abs(fx) > 0.001 or abs(fy) > 0.001 or abs(fz) > 0.001:
            # Draw force arrow
            self.ax_force.quiver(0, 0, 0, fx, fy, fz,
                               color='red', arrow_length_ratio=0.1, linewidth=6,
                               alpha=0.8)

        # Update text (always show, display original values)
        magnitude = np.sqrt(fx**2 + fy**2 + fz**2)
        self.force_text.set_text(
            f"FORCE (N)\n"
            f"Fx: {fx:7.3f}\n"
            f"Fy: {fy:7.3f}\n"
            f"Fz: {fz:7.3f}\n"
            f"|F|: {magnitude:7.3f}"
        )

        # Update Torque plot
        self.ax_torque.cla()
        self._setup_axis(self.ax_torque, "Torque (Nm)", self.TORQUE_LIMIT)

        if abs(mx) > 0.001 or abs(my) > 0.001 or abs(mz) > 0.001:
            # Draw torque arrow
            self.ax_torque.quiver(0, 0, 0, mx, my, mz,
                                color='purple', arrow_length_ratio=0.1, linewidth=6,
                                alpha=0.8)

        # Update text (always show)
        magnitude = np.sqrt(mx**2 + my**2 + mz**2)
        self.torque_text.set_text(
            f"TORQUE (Nm)\n"
            f"Mx: {mx:7.3f}\n"
            f"My: {my:7.3f}\n"
            f"Mz: {mz:7.3f}\n"
            f"|M|: {magnitude:7.3f}"
        )

        return []


def main():
    parser = argparse.ArgumentParser(description="Parse force sensor data with optional 3D visualization")
    parser.add_argument("--port", default="/dev/ttyACM0", help="Serial port")
    parser.add_argument("--baudrate", type=int, default=115200, help="Baudrate")
    parser.add_argument("--duration", type=float, default=None, help="Duration in seconds")
    parser.add_argument("--no-stats", action="store_true", help="Disable statistics")
    parser.add_argument("--single", action="store_true", help="Read single sample and exit")
    parser.add_argument("--vis", action="store_true", help="Enable 3D visualization")

    args = parser.parse_args()

    # Check if visualization is requested but matplotlib is not available
    if args.vis and not MATPLOTLIB_AVAILABLE:
        print("Error: --vis requires matplotlib. Install with: pip install matplotlib")
        return

    reader = ForceSensorReader(args.port, args.baudrate)

    try:
        reader.connect()

        # Ask user if they want to tare
        print("\n" + "="*50)
        tare_input = input("Do you want to tare (zero) the sensor? (y/n): ").strip().lower()
        if tare_input == 'y' or tare_input == 'yes':
            reader.tare(duration=10.0)
        else:
            print("[Tare] Skipped. Using raw values.\n")
        print("="*50 + "\n")

        # Ask user for output filename
        print("="*50)
        print(f"Data will be saved to: {OUTPUT_DIR}/")
        filename_input = input("Enter filename (without .csv, or press Enter to skip saving): ").strip()
        if filename_input:
            # Create output directory if it doesn't exist
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            # Add .csv extension if not present
            if not filename_input.endswith('.csv'):
                filename_input += '.csv'
            output_path = os.path.join(OUTPUT_DIR, filename_input)
            reader.output_file = output_path
            print(f"[Recording] Will save to: {output_path}")
        else:
            print("[Recording] Skipped. Data will not be saved.")
        print("="*50 + "\n")

        # Start recording if output file specified
        reader.start_recording()

        if args.single:
            data = reader.read_single()
            data = reader.apply_offset(data)  # Apply tare offset
            reader.save_data(data)  # Save to file if recording
            print(f"Force: ({data['fx']}, {data['fy']}, {data['fz']})")
            print(f"Torque: ({data['mx']}, {data['my']}, {data['mz']})")
        elif args.vis:
            # 3D visualization mode
            print("Starting 3D visualization mode...")
            print("Data will be printed to terminal.")
            print("Close the plot window to exit.\n")

            visualizer = ForceSensor3DVisualizer(reader)
            visualizer.init_plot()

            # Data container for visualization callback
            latest_data = {'fx': 0, 'fy': 0, 'fz': 0, 'mx': 0, 'my': 0, 'mz': 0}
            count = [0]  # Use list to allow modification in nested function
            last_stats_time = [time.time()]
            data_buffer = {k: [] for k in ['fx', 'fy', 'fz', 'mx', 'my', 'mz']}

            def update_callback(frame):
                """Callback for animation that reads data and updates plot."""
                # Read ALL waiting data packets (not just one)
                while reader.ser.in_waiting > 0:
                    line = reader.ser.readline().decode('utf-8', errors='ignore').strip()
                    data = reader.parse_line(line)

                    if data:
                        # Apply tare offset
                        data = reader.apply_offset(data)
                        count[0] += 1
                        latest_data.update(data)

                        # Save to file if recording
                        reader.save_data(data)

                        # Print to terminal (SI units: N, Nm)
                        print(f"[{count[0]:4d}] Fx:{data['fx']:7.3f}N Fy:{data['fy']:7.3f}N Fz:{data['fz']:7.3f}N "
                              f"Mx:{data['mx']:6.3f}Nm My:{data['my']:6.3f}Nm Mz:{data['mz']:6.3f}Nm")

                        # Store for statistics
                        if not args.no_stats:
                            for key in data_buffer:
                                data_buffer[key].append(data[key])

                # Print statistics every second
                if not args.no_stats and (time.time() - last_stats_time[0]) >= 1.0:
                    reader._print_stats(data_buffer)
                    data_buffer.clear()
                    for k in ['fx', 'fy', 'fz', 'mx', 'my', 'mz']:
                        data_buffer[k] = []
                    last_stats_time[0] = time.time()

                # Update visualization with latest data
                return visualizer.update_plot(frame, lambda: latest_data)

            # Create animation
            anim = FuncAnimation(visualizer.fig, update_callback,
                               interval=50, cache_frame_data=False)

            plt.show()
        else:
            # Text-only mode
            reader.read_continuous(
                duration=args.duration,
                show_stats=not args.no_stats
            )

    except serial.SerialException as e:
        print(f"Serial error: {e}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        reader.stop_recording()
        reader.disconnect()


if __name__ == "__main__":
    main()
