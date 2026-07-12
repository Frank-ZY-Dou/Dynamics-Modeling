#!/usr/bin/env python
"""
Real-time 3D visualization of force sensor data.
Displays current force vector in 3D coordinate system.
"""

import serial
import time
import re
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation


class ForceSensor3DPlotter:
    """Real-time 3D plotter for force sensor."""

    def __init__(self, port="/dev/ttyACM0", baudrate=115200, print_data=True):
        self.port = port
        self.baudrate = baudrate
        self.print_data = print_data
        self.ser = None

        # Current force values
        self.fx = 0
        self.fy = 0
        self.fz = 0
        self.mx = 0
        self.my = 0
        self.mz = 0

        self.count = 0

        # For statistics
        self.last_stats_time = None
        self.data_buffer = {k: [] for k in ['fx', 'fy', 'fz', 'mx', 'my', 'mz']}

    def connect(self):
        """Connect to sensor."""
        print(f"Connecting to {self.port} at {self.baudrate} baud...")
        self.ser = serial.Serial(
            port=self.port,
            baudrate=self.baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=0.01
        )
        print("Connected!")

    def disconnect(self):
        """Disconnect from sensor."""
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("Disconnected")

    def parse_line(self, line):
        """Parse sensor data line."""
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

    def read_data(self):
        """Read and parse new data from sensor."""
        if not self.ser or not self.ser.is_open:
            return None

        if self.ser.in_waiting > 0:
            line = self.ser.readline().decode('utf-8', errors='ignore').strip()
            data = self.parse_line(line)

            if data:
                self.count += 1
                self.fx = data['fx']
                self.fy = data['fy']
                self.fz = data['fz']
                self.mx = data['mx']
                self.my = data['my']
                self.mz = data['mz']

                # Print data to terminal
                if self.print_data:
                    print(f"[{self.count:4d}] Fx:{self.fx:6d} Fy:{self.fy:6d} Fz:{self.fz:6d} "
                          f"Mx:{self.mx:4d} My:{self.my:4d} Mz:{self.mz:4d}")

                # Store for statistics
                for key in self.data_buffer:
                    self.data_buffer[key].append(data[key])

                # Print statistics every second
                if self.last_stats_time is None:
                    self.last_stats_time = time.time()

                if (time.time() - self.last_stats_time) >= 1.0:
                    self._print_stats()
                    # Clear buffer
                    self.data_buffer = {k: [] for k in ['fx', 'fy', 'fz', 'mx', 'my', 'mz']}
                    self.last_stats_time = time.time()

                return data
        return None

    def _print_stats(self):
        """Print statistics for the last second."""
        if not self.data_buffer['fx']:
            return

        print("\n--- Stats (last 1s) ---")
        for key in ['fx', 'fy', 'fz', 'mx', 'my', 'mz']:
            values = self.data_buffer[key]
            print(f"{key.upper()}: mean={np.mean(values):7.1f} std={np.std(values):6.1f} "
                  f"min={np.min(values):6d} max={np.max(values):6d}")
        print()

    def init_plot(self):
        """Initialize the 3D plot."""
        self.fig = plt.figure(figsize=(14, 7))

        # Create two subplots: Force (left) and Torque (right)
        self.ax_force = self.fig.add_subplot(121, projection='3d')
        self.ax_torque = self.fig.add_subplot(122, projection='3d')

        # Setup Force plot
        self._setup_axis(self.ax_force, "Force (ADC counts)")

        # Setup Torque plot
        self._setup_axis(self.ax_torque, "Torque (ADC counts)")

        # Create arrow objects (quiver)
        self.force_arrow = None
        self.torque_arrow = None

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

    def _setup_axis(self, ax, title):
        """Setup a 3D axis."""
        # Set labels
        ax.set_xlabel('X', fontsize=12, fontweight='bold', color='red')
        ax.set_ylabel('Y', fontsize=12, fontweight='bold', color='green')
        ax.set_zlabel('Z', fontsize=12, fontweight='bold', color='blue')
        ax.set_title(title, fontsize=14, fontweight='bold')

        # Set initial limits
        limit = 10000
        ax.set_xlim([-limit, limit])
        ax.set_ylim([-limit, limit])
        ax.set_zlim([-limit, limit])

        # Draw coordinate axes
        ax.plot([0, limit*0.8], [0, 0], [0, 0], 'r-', alpha=0.3, linewidth=1)
        ax.plot([0, 0], [0, limit*0.8], [0, 0], 'g-', alpha=0.3, linewidth=1)
        ax.plot([0, 0], [0, 0], [0, limit*0.8], 'b-', alpha=0.3, linewidth=1)

        # Grid
        ax.grid(True, alpha=0.2)

    def update_plot(self, frame):
        """Update function for animation."""
        # Read new data
        self.read_data()

        # Update Force plot
        self.ax_force.cla()
        self._setup_axis(self.ax_force, "Force (ADC counts)")

        if self.fx != 0 or self.fy != 0 or self.fz != 0:
            # Draw force arrow
            self.ax_force.quiver(0, 0, 0, self.fx, self.fy, self.fz,
                               color='red', arrow_length_ratio=0.1, linewidth=3,
                               alpha=0.8)

            # Draw magnitude
            magnitude = np.sqrt(self.fx**2 + self.fy**2 + self.fz**2)
            self.force_text.set_text(
                f"FORCE\n"
                f"Fx: {self.fx:7d}\n"
                f"Fy: {self.fy:7d}\n"
                f"Fz: {self.fz:7d}\n"
                f"|F|: {magnitude:7.0f}\n"
                f"Samples: {self.count}"
            )

        # Update Torque plot
        self.ax_torque.cla()
        self._setup_axis(self.ax_torque, "Torque (ADC counts)")

        if self.mx != 0 or self.my != 0 or self.mz != 0:
            # Draw torque arrow
            self.ax_torque.quiver(0, 0, 0, self.mx, self.my, self.mz,
                                color='purple', arrow_length_ratio=0.1, linewidth=3,
                                alpha=0.8)

            # Draw magnitude
            magnitude = np.sqrt(self.mx**2 + self.my**2 + self.mz**2)
            self.torque_text.set_text(
                f"TORQUE\n"
                f"Mx: {self.mx:5d}\n"
                f"My: {self.my:5d}\n"
                f"Mz: {self.mz:5d}\n"
                f"|M|: {magnitude:5.0f}"
            )

        return []

    def run(self):
        """Start real-time 3D plotting."""
        try:
            self.connect()
            self.init_plot()

            print("Starting real-time 3D visualization...")
            if self.print_data:
                print("Data will be printed to terminal with statistics every second.")
            print("Close the plot window to exit.\n")

            # Create animation (update every 50ms = 20Hz)
            anim = FuncAnimation(self.fig, self.update_plot,
                               interval=50, cache_frame_data=False)

            plt.show()

        except KeyboardInterrupt:
            print("\nStopped by user")
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.disconnect()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Real-time 3D force sensor visualization")
    parser.add_argument("--port", default="/dev/ttyACM0", help="Serial port")
    parser.add_argument("--baudrate", type=int, default=115200, help="Baudrate")
    parser.add_argument("--no-print", action="store_true", help="Disable terminal data printing")

    args = parser.parse_args()

    plotter = ForceSensor3DPlotter(args.port, args.baudrate, print_data=not args.no_print)
    plotter.run()


if __name__ == "__main__":
    main()
