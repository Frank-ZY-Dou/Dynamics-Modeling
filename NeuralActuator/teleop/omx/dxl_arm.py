#!/usr/bin/env python

import rospy
import threading
import csv
import matplotlib.pyplot as plt
import numpy as np
from finger_control.hand_control import HandControlDXL
from math import pi, sqrt, sin, cos
from finger_control.force_reader_class import ForceGaugeReader

from soft_hand_control.msg import MotorMonitorNoLength
import subprocess
import signal
import os
import serial
import re
import time as time_module


# =============================================================================
# PID Configuration for Extended Position Control (Mode 4)
# Used by record_data() for hardware PID setup
# =============================================================================
PID_CONFIG = {
    # Motor-specific PID gains: {motor_id: {'kp': value, 'ki': value, 'kd': value}}
    11: {'kp': 800, 'ki': 0, 'kd': 1000},   # base
    12: {'kp': 800, 'ki': 0, 'kd': 800},   # shoulder
    13: {'kp': 800, 'ki': 0, 'kd': 500},   # elbow
    14: {'kp': 800, 'ki': 0, 'kd': 300},   # wrist
    15: {'kp': 800, 'ki': 0, 'kd': 0},   # gripper
}



class ForceSensorReader:
    """
    Reader for 6-axis force/torque sensor.
    Reads fx, fy, fz (force) and mx, my, mz (torque) from serial port.
    """

    def __init__(self, port="/dev/ttyACM0", baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        # Tare offset (initialized to zero)
        self.offset = {'fx': 0, 'fy': 0, 'fz': 0, 'mx': 0, 'my': 0, 'mz': 0}
        self.connected = False

    def connect(self):
        """Connect to sensor."""
        try:
            print(f"[ForceSensor] Connecting to {self.port}...")
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.1
            )
            self.connected = True
            print("[ForceSensor] Connected!")
            return True
        except serial.SerialException as e:
            print(f"[ForceSensor] Connection failed: {e}")
            self.connected = False
            return False

    def disconnect(self):
        """Disconnect from sensor."""
        if self.ser and self.ser.is_open:
            self.ser.close()
            self.connected = False
            print("[ForceSensor] Disconnected")

    def parse_line(self, line):
        """
        Parse sensor data line.
        Format: < val1 val2 val3 val4 val5 val6 >
        """
        pattern = r'<\s*(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)\s*>'
        match = re.search(pattern, line)
        if match:
            values = [int(match.group(i)) for i in range(1, 7)]
            return {
                'fx': values[0], 'fy': values[1], 'fz': values[2],
                'mx': values[3], 'my': values[4], 'mz': values[5]
            }
        return None

    def apply_offset(self, data):
        """Apply tare offset and convert to SI units (N, Nm)."""
        if data is None:
            return None
        return {
            'fx': (data['fx'] - self.offset['fx']) / 1000.0,
            'fy': (data['fy'] - self.offset['fy']) / 1000.0,
            'fz': (data['fz'] - self.offset['fz']) / 1000.0,
            'mx': (data['mx'] - self.offset['mx']) / 1000.0,
            'my': (data['my'] - self.offset['my']) / 1000.0,
            'mz': (data['mz'] - self.offset['mz']) / 1000.0,
        }

    def tare(self, duration=5.0):
        """Perform tare (zero) operation."""
        if not self.ser or not self.ser.is_open:
            print("[ForceSensor] Not connected, cannot tare")
            return False

        print(f"[ForceSensor] Taring for {duration:.1f}s - keep sensor unloaded!")
        tare_buffer = {k: [] for k in ['fx', 'fy', 'fz', 'mx', 'my', 'mz']}
        start_time = time_module.time()

        while (time_module.time() - start_time) < duration:
            if self.ser.in_waiting > 0:
                line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                data = self.parse_line(line)
                if data:
                    for key in tare_buffer:
                        tare_buffer[key].append(data[key])
                    elapsed = time_module.time() - start_time
                    print(f"\r[ForceSensor] Tare: {elapsed:.1f}s / {duration:.1f}s", end='')
            else:
                time_module.sleep(0.001)

        print()  # New line

        if not tare_buffer['fx']:
            print("[ForceSensor] Tare failed: no data!")
            return False

        for key in self.offset:
            self.offset[key] = int(np.mean(tare_buffer[key]))

        print(f"[ForceSensor] Tare done. Offset: Fx={self.offset['fx']} Fy={self.offset['fy']} Fz={self.offset['fz']}")
        return True

    def read_single(self):
        """Read a single data point (non-blocking, returns None if no data)."""
        if not self.ser or not self.ser.is_open:
            return None
        if self.ser.in_waiting > 0:
            line = self.ser.readline().decode('utf-8', errors='ignore').strip()
            return self.parse_line(line)
        return None

    def read_latest(self):
        """Read all waiting data and return the latest (to avoid lag)."""
        if not self.ser or not self.ser.is_open:
            return None
        latest = None
        while self.ser.in_waiting > 0:
            line = self.ser.readline().decode('utf-8', errors='ignore').strip()
            data = self.parse_line(line)
            if data:
                latest = data
        return latest


class VideoRecorder:
    """
    Video recorder using ffmpeg with libx264 CPU encoding.
    Records from USB camera at configurable resolution with audio.

    Resolution presets:
        "4k"    -> 3840x2160 (highest quality, may not work on all cameras)
        "qhd"   -> 2560x1440 (good balance between quality and compatibility)
        "1080p" -> 1920x1080 (most compatible)
        Or specify custom like "2560x1440"
    """

    # Resolution presets
    RESOLUTION_PRESETS = {
        "4k": "3840x2160",
        "2k": "2560x1440",
        "qhd": "2560x1440",
        "1080p": "1920x1080",
        "720p": "1280x720",
    }

    def __init__(self, device="/dev/video1", resolution="qhd", fps=30,
                 audio_device="default", record_audio=True):
        """
        Args:
            device: Video device path (default: /dev/video1)
            resolution: Recording resolution - preset name or "WxH" (default: qhd)
            fps: Frames per second (default: 30)
            audio_device: Audio device for recording (default: "default" for system default mic)
            record_audio: Whether to record audio (default: True)
        """
        self.device = device
        # Support preset names like "qhd" or direct resolution like "2560x1440"
        self.resolution = self.RESOLUTION_PRESETS.get(resolution.lower(), resolution)
        self.fps = fps
        self.audio_device = audio_device
        self.record_audio = record_audio
        self.process = None
        self.output_path = None
        self.is_recording = False
        self.stderr_thread = None
        self.stderr_output = ""

    def _read_stderr(self):
        """Background thread to read stderr and prevent blocking."""
        try:
            for line in iter(self.process.stderr.readline, b''):
                self.stderr_output += line.decode('utf-8', errors='replace')
        except:
            pass

    def start(self, output_path):
        """
        Start video recording.

        Args:
            output_path: Path to save the video file (e.g., /path/to/video.mp4)
        """
        if self.is_recording:
            print("[VideoRecorder] Already recording!")
            return False

        self.output_path = output_path
        self.stderr_output = ""

        # Create output directory if needed
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"[VideoRecorder] Created directory: {output_dir}")

        # Build ffmpeg command with libx264 CPU encoding and optional audio
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output
            # Video input
            "-f", "v4l2",
            "-video_size", self.resolution,
            "-framerate", str(self.fps),
            "-thread_queue_size", "1024",
            "-i", self.device,
        ]

        # Add audio input if enabled
        if self.record_audio:
            cmd.extend([
                "-f", "pulse",  # PulseAudio (common on Ubuntu)
                "-i", self.audio_device,
            ])

        # Video encoding
        cmd.extend([
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
        ])

        # Audio encoding if enabled
        if self.record_audio:
            cmd.extend([
                "-c:a", "aac",
                "-b:a", "128k",
            ])

        cmd.append(output_path)

        print(f"[VideoRecorder] Starting recording: {output_path}")
        print(f"[VideoRecorder] Resolution: {self.resolution} @ {self.fps}fps")
        print(f"[VideoRecorder] Audio: {'enabled' if self.record_audio else 'disabled'}")
        print(f"[VideoRecorder] Command: {' '.join(cmd)}")

        try:
            # Start ffmpeg process
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE
            )

            # Start background thread to read stderr (prevents blocking)
            self.stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
            self.stderr_thread.start()

            # Wait briefly and check if process is still running
            import time
            time.sleep(0.5)
            if self.process.poll() is not None:
                # Process exited immediately - likely an error
                time.sleep(0.2)  # Give stderr thread time to read
                print(f"[VideoRecorder] ERROR: ffmpeg exited immediately!")
                print(f"[VideoRecorder] ffmpeg stderr:\n{self.stderr_output}")
                return False

            self.is_recording = True
            print(f"[VideoRecorder] Recording started successfully (libx264 CPU)")
            return True
        except Exception as e:
            print(f"[VideoRecorder] Failed to start recording: {e}")
            return False

    def stop(self):
        """
        Stop video recording gracefully.
        """
        if not self.is_recording or self.process is None:
            print("[VideoRecorder] Not recording")
            return False

        print("[VideoRecorder] Stopping recording...")

        try:
            # Check if process is still running
            if self.process.poll() is not None:
                # Process already exited
                import time
                time.sleep(0.2)  # Give stderr thread time to finish
                print(f"[VideoRecorder] ffmpeg already exited with code {self.process.returncode}")
                if self.process.returncode != 0 and self.stderr_output:
                    print(f"[VideoRecorder] ffmpeg stderr:\n{self.stderr_output[-2000:]}")
                self.is_recording = False
                return self.process.returncode == 0

            # Method 1: Try sending 'q' to stdin (graceful quit)
            try:
                self.process.stdin.write(b'q')
                self.process.stdin.flush()
            except (BrokenPipeError, OSError):
                # stdin pipe broken, try SIGINT instead
                pass

            # Wait briefly for graceful exit
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Method 2: Send SIGINT (like Ctrl+C)
                print("[VideoRecorder] Sending SIGINT...")
                self.process.send_signal(signal.SIGINT)
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # Method 3: Force kill
                    print("[VideoRecorder] Force killing ffmpeg...")
                    self.process.kill()
                    self.process.wait()

            # Give stderr thread time to finish reading
            import time
            time.sleep(0.3)

            # Display any stderr output (useful for debugging)
            if self.process.returncode != 0:
                print(f"[VideoRecorder] ffmpeg exited with code {self.process.returncode}")
                if self.stderr_output:
                    print(f"[VideoRecorder] ffmpeg stderr:\n{self.stderr_output[-2000:]}")

            self.is_recording = False
            print(f"[VideoRecorder] Recording saved: {self.output_path}")
            return True

        except Exception as e:
            print(f"[VideoRecorder] Error stopping recording: {e}")
            # Try to kill anyway
            try:
                self.process.kill()
                self.process.wait()
            except:
                pass
            self.is_recording = False
            return False

    def __del__(self):
        """Ensure recording is stopped on cleanup."""
        if self.is_recording:
            self.stop()


rospy.init_node("arm_control_node")

class DXL_ARM:

    # Temperature thresholds (Dynamixel XM430 max operating temp is 80°C)
    TEMP_WARNING = 55    # Warning threshold (°C)
    TEMP_CRITICAL = 60   # Critical threshold - auto disable (°C)

    def __init__(self, siblings=False, twin=False, force_gauge=None,
                 force_sensor=None, force_sensor_direction=None, device=None):
        """
        Args:
            siblings: If True, use twin arm mode
            twin: Reference to the other arm in twin mode
            force_gauge: Legacy single-axis force gauge reader
            force_sensor: ForceSensorReader instance for 6-axis force/torque sensor
            force_sensor_direction: Direction of applied force ('x', 'y', or 'z')
                - 'x': sensor XYZ -> gripper XYZ
                - 'y': sensor XYZ -> gripper YXZ (swap X and Y)
                - 'z': sensor XYZ -> gripper XYZ
            device: USB device path (e.g., '/dev/ttyUSB0' or '/dev/ttyUSB1')
                    If None, auto-select based on siblings/twin mode.
        """
        self.device = device  # Store for reference
        self.arm_status = None
        self.siblings = siblings
        self.twin = twin
        self.rate = rospy.Rate(100) # Rate in Hz
        self.lock = threading.Lock() # This solves a large error, assures commands dont get mixed up if multiple are sent at the same time
        self.force_vector = [0,0,1]
        self.csv_store = 0

        # 6-axis force sensor
        self.force_sensor = force_sensor
        self.force_sensor_direction = force_sensor_direction
        self.force_sensor_data = {'fx': 0, 'fy': 0, 'fz': 0}  # Latest reading
        self.force_sensor_lock = threading.Lock()
        self.live_current = 0 # If the value is one, the data won't record
        self.temp_warning_issued = False  # Avoid repeated warnings
        self.temp_critical_issued = False  # Avoid repeated critical alerts
        self.goal_pos = [0.0, 0.0, 0.0, 0.0, 0.0]  # Goal position from leader arm (for CSV recording)
        # Event to signal when recording actually starts (after user presses Enter in twin mode)
        self.recording_started_event = threading.Event()
        self.recording_start_time = None  # Set by record_data() when recording starts
        self.ap_length = 25
        self.ap_radius = 15
        self.ap_delta = 2.5
        self.l_12 = 0.130
        self.l_13 = 0.125
        self.l_grip = 0.148
        self.modes = {"current": 0, "velocity":1, "position":3, "ext_position":4, "current_pos":5}
        arm_setup = {}
        self.home_offset = [-2048, -2048, -2048, -2048, 2048]
        # Gripper offset for twin following mode (applied to gripper position when following)
        # Calibration: follower gripper is ~0.35 rad wider than leader when closed
        # Negative offset makes follower close more to match leader
        self.gripper_follow_offset = -0.35  # radians (adjust based on calibration)
        self.default = [0, 0, 0, 0, 0]
        self.downward = [-0.02300980999305504, 0.6933622744573918, 0.6626825277999852, 1.5769389781907055, 1.2732094862823788]
        self.upward = [0, 0, -pi/2, 0, 1.2732094862823788]
        self.left = [1.612220686846723, 0.09510721463796083, 0.4095746178763797, -0.2883896185796232, 1.4035984095763574]
        self.motor_limits = [(-pi,pi), (-1.5,1.5), (-1.5,1.3820), (-1.7, 1.97),(0,pi/2)]
        if self.twin is None and self.siblings is True:
                    # All 5 motors
                    dxl_arm_config = {'motor_ids': [11, 14, 13, 12, 15],
                        'motor_dirs': [0, 0, 0, 0, 1],
                        'motor_limits': None,
                        'forward_model': None,
                        'backward_model': None,
                        'forward_model_const': None,
                        'backward_model_const': None,
                        'home_offset':self.home_offset}
        else:
            # All 5 motors
            dxl_arm_config = {'motor_ids': [11, 12, 13, 14, 15],
                            'motor_dirs': [0, 0, 0, 0, 1],
                            'motor_limits': None,
                            'forward_model': None,
                            'backward_model': None,
                            'forward_model_const': None,
                            'backward_model_const': None,
                            'home_offset':self.home_offset}
        arm_setup['arm'] = dxl_arm_config

        # Determine USB device path
        if self.device is not None:
            # Use explicitly specified device
            usb_device = self.device
        elif self.twin is None and self.siblings is True:
            # Leader arm (twin mode) - USB0
            usb_device = '/dev/ttyUSB0'
        elif self.siblings is True:
            # Follower arm (twin mode) - USB1
            usb_device = '/dev/ttyUSB1'
        else:
            # Single arm mode - USB0
            usb_device = '/dev/ttyUSB0'

        print(f"Using USB device: {usb_device}")
        self.dxl_arm = HandControlDXL(arm_setup, usb_device,
                                init_motors=False, init_all_motors=False)
        
        self.force_gauge = force_gauge
        with self.lock:
            self.dxl_arm.switch_motor_operating_mode(4)  # Extended Position Control
        print("All motors switched to: ext_position (Extended Position Control)")

        # Set PID immediately after switching to Mode 4
        # Disable all motors first to avoid jerking during PID setup
        with self.lock:
            self.dxl_arm.fingers["arm"].deactive_motors(None)

        # Set PID gains (motors already disabled, so no jerking)
        packetHandler = self.dxl_arm.fingers["arm"]._FingerControlDXL__dxl_packetHandler
        portHandler = self.dxl_arm.fingers["arm"]._FingerControlDXL__dxl_portHandler
        packetHandler.write2ByteTxRx(portHandler, 11, 84, 50)   # Kp
        packetHandler.write2ByteTxRx(portHandler, 11, 80, 800)  # Kd
        packetHandler.write2ByteTxRx(portHandler, 12, 84, 100)  # Kp
        packetHandler.write2ByteTxRx(portHandler, 12, 80, 600)  # Kd
        packetHandler.write2ByteTxRx(portHandler, 13, 84, 180)  # Kp
        packetHandler.write2ByteTxRx(portHandler, 13, 80, 650)  # Kd
        packetHandler.write2ByteTxRx(portHandler, 14, 84, 120)  # Kp
        packetHandler.write2ByteTxRx(portHandler, 14, 80, 450)  # Kd
        packetHandler.write2ByteTxRx(portHandler, 15, 84, 130)  # Kp
        packetHandler.write2ByteTxRx(portHandler, 15, 80, 300)  # Kd

        # Re-enable all motors at once (write torque enable = 1)
        for motor_id in [11, 12, 13, 14, 15]:
            packetHandler.write1ByteTxRx(portHandler, motor_id, 64, 1)  # 64 = TORQUE_ENABLE address
        print("PID gains set for Mode 4")
        rospy.sleep(0.5)

    def start_node(self):
        rospy.init_node("arm_control_node")

    def switch_mode(self, mode, id=None):
        ## MODE SHOULD BE A STRING ##

        # if id:
        #     if id not in modes:
        #      print("mode doesn't exist")
        if id:
            with self.lock:
                self.dxl_arm.fingers["arm"].switch_single_motor_operating_mode(id, self.modes[mode])
            print(f"Motor {id} switched to:", mode)
        else:
            with self.lock:
                self.dxl_arm.fingers["arm"].switch_motor_operating_mode(self.modes[mode])
            print("All motors switched to:", mode)

    def get_pos(self):
        
        return self.dxl_arm.fingers["arm"].get_motor_pos()

    def set_velocity(self, profile):
        with self.lock:
            self.dxl_arm.fingers["arm"].set_velocity_profile(profile)
        print("velocity set")

    def set_pid_gains(self, kp=800, ki=100, kd=1000):
        """
        Set PID gains for position control on all motors.

        Args:
            kp: Position P Gain (default: 800)
            ki: Position I Gain (default: 100)
            kd: Position D Gain (default: 1000)
        """
        with self.lock:
            self.dxl_arm.fingers["arm"].set_pid_gains(kp, ki, kd)

    def get_pid_gains(self):
        """
        Read current PID gains from all motors for verification.

        Returns:
            dict: {motor_id: {'kp': value, 'ki': value, 'kd': value}, ...}
        """
        with self.lock:
            return self.dxl_arm.fingers["arm"].get_pid_gains()

    def set_single_motor_pid_gains(self, motor_id, kp, ki, kd):
        """
        Set PID gains for a single motor.

        Args:
            motor_id: Motor ID (e.g., 11, 12, 13, 14)
            kp: Position P Gain
            ki: Position I Gain
            kd: Position D Gain
        """
        with self.lock:
            self.dxl_arm.fingers["arm"].set_single_motor_pid_gains(motor_id, kp, ki, kd)

    def set_goal_current(self, goal_current_mA):
        """
        Set Goal Current for all motors (used in Current-Based Position Control Mode).

        Args:
            goal_current_mA: Goal current in mA
        """
        with self.lock:
            self.dxl_arm.fingers["arm"].set_goal_current(goal_current_mA)

    def get_goal_current(self):
        """
        Read Goal Current from all motors.
        """
        with self.lock:
            self.dxl_arm.fingers["arm"].get_goal_current()

    def move_when_position(self, id, pos_to_move, id_pos, pos, precision = 0):
        with self.lock:
            start_pos = self.dxl_arm.fingers['arm'].get_motor_pos()[id_pos % 11]
    
        sign = (pos - start_pos) > precision
    
        start = rospy.Time.now().to_sec()
        with self.lock:
            current_pos = self.dxl_arm.fingers['arm'].get_motor_pos()[id_pos % 11]

        while ((pos - current_pos)> precision) == sign:
            with self.lock:
                current_pos = self.dxl_arm.fingers['arm'].get_motor_pos()[id_pos % 11]

            limit = 5

            if (rospy.Time.now().to_sec() - start) > limit:
                print(f"Motor didn't hit specified end position in {limit} seconds")
                return
            self.rate.sleep()
        with self.lock:
            self.dxl_arm.fingers['arm'].send_single_motor_pos_cmd(pos_to_move, id)



    def torque_till_position(self, id, torque, pos = None, id_pos = None):
        
        ##### DOESN'T WORK THAT WELL - A LITTLE SLOW TO STOP MOTORS ######

        if id_pos is None:
            # MONITORING POSITION OF MOTOR THAT IS HAVING TORQUE APPLIED
            id_pos = id

        with self.lock:
            # print("torque applied")
            self.dxl_arm.fingers['arm'].send_motor_torque_cmd(id, torque) # APPLY TORQUE 

        if pos != None:
            with self.lock:
                current_pos = self.dxl_arm.fingers['arm'].get_motor_pos()[id_pos % 11]

            sign = (pos - current_pos) > 0
            
            start = rospy.Time.now().to_sec()
            while (((pos - current_pos) > 0) == sign):
                with self.lock:
                    current_pos = self.dxl_arm.fingers['arm'].get_motor_pos()[id_pos % 11]

                if (rospy.Time.now().to_sec() - start) > 5:
                    print(f"Motor {id} never hit specified end position")
                    break
                self.rate.sleep()
                
            with self.lock:
                self.dxl_arm.fingers['arm'].send_motor_torque_cmd(id, 0) # STOP APPLYING TORQUE 

    def torque_for_time(self, id, torque, time, all=None):
        start = rospy.Time.now().to_sec()
        if all==True:
            with self.lock:
                self.dxl_arm.fingers['arm'].send_torque_sync_cmd(torque)
        else:
            with self.lock:
                self.dxl_arm.fingers['arm'].send_motor_torque_cmd(id, torque)

        while (rospy.Time.now().to_sec() - start) < time:
            self.rate.sleep()
        if all==True:
            with self.lock:
                self.dxl_arm.fingers['arm'].send_torque_sync_cmd([0,0,0,0,0])
        else:
            with self.lock:
                self.dxl_arm.fingers['arm'].send_motor_torque_cmd(id, 0)





    ####### ACTION FUNCTIONS ########

    def press_enter(self, csv = None):
        if csv != None:
            self.csv(self.data[csv:], f"arm_data_{self.csv_store}.csv")
            print("csv", self.csv_store)
            self.csv_store +=1
        command = input("Press Enter to continue, press x to exit...")
        if command == "x":
            self.dxl_arm.fingers['arm'].deactive_motors(None)  # None = all motors
        

    def rest(self, time = None):
        if time:
            rospy.sleep(time)
        else:
            rospy.sleep(1.0)

    # box
    def box(self, ):
        self.dxl_arm.fingers['arm'].send_motor_pos_cmd([0.0, 0.0, 2.0, 1.0, 1.65])
        print("box")
        self.rest()
        self.press_enter()
    # ball
    def ball(self):
        self.dxl_arm.fingers['arm'].send_motor_pos_cmd([0.0, 0.0, 2.0, 1.0, 1.2])
        print("ball")
        self.rest()
        self.press_enter()
    # soft cube
    def soft_cube(self):
        self.dxl_arm.fingers['arm'].send_motor_pos_cmd([0.0, 0.0, 2.0, 1.0, 0.6])
        print("soft cube")
        self.rest()
        self.press_enter()
    # banana
    def banana(self):
        self.dxl_arm.fingers['arm'].send_motor_pos_cmd([0.0, 0.0, 2.0, 1.0, 0.65])
        print("banana")
        self.rest()
        self.press_enter()

    def home(self):
        self.dxl_arm.fingers['arm'].send_motor_pos_cmd(self.default)
        print("home")
        self.rest()
        self.press_enter()

    def down(self):
        self.dxl_arm.fingers['arm'].send_motor_pos_cmd(self.downward)
        print("down")
        self.rest()
        self.press_enter()

    def up(self,id = None, pause = True):
        
        if id:
            with self.lock:
                if id == 13:
                    self.dxl_arm.fingers['arm'].send_single_motor_pos_cmd(-pi/2, id)
                else:
                    self.dxl_arm.fingers['arm'].send_single_motor_pos_cmd(0, id)
        else:
            with self.lock:
                self.dxl_arm.fingers['arm'].send_motor_pos_cmd(self.upward)
        print("up", id)
        
        if pause:
            self.rest()
            self.press_enter()

    def move_left(self):
        # dxl_arm.fingers['arm'].send_motor_pos_cmd(downward)
        self.dxl_arm.fingers['arm'].send_motor_pos_cmd(self.left)
        print("left")
        self.rest()
        self.press_enter()

    def default_move(self):
        self.down()
        self.up()
        self.move_left()
        self.switch_mode("current", 15)
        self.dxl_arm.fingers['arm'].send_motor_torque_cmd(15, -300)
        self.rest(4.0)
        self.switch_mode("ext_position")
        self.up()

    def disable(self,id=None):
        self.switch_mode("ext_position")
        if id:
            print(f"Disabling motor {id}...")
        else:
            print("Disabling all motors...")
        with self.lock:
            self.dxl_arm.fingers['arm'].deactive_motors(id)
        rospy.sleep(3)
        print(f"Motors disabled")
        self.press_enter()

    def check_temperature(self, temps, motor_ids=[11, 12, 13, 14, 15]):
        """
        Check motor temperatures and warn/disable if too hot.

        Args:
            temps: List of temperature values for each motor
            motor_ids: List of motor IDs corresponding to temps

        Returns:
            True if critical (should stop), False otherwise
        """
        if not temps or len(temps) == 0:
            return False

        max_temp = max(temps)
        max_idx = temps.index(max_temp)
        max_motor = motor_ids[max_idx] if max_idx < len(motor_ids) else "?"

        # Critical threshold - auto disable
        if max_temp >= self.TEMP_CRITICAL:
            if not self.temp_critical_issued:
                print("\n" + "!"*60)
                print(f"!!! CRITICAL: Motor {max_motor} temperature {max_temp}°C !!!")
                print(f"!!! Exceeds critical threshold ({self.TEMP_CRITICAL}°C) !!!")
                print("!!! AUTO-DISABLING MOTORS - POWER OFF IMMEDIATELY !!!")
                print("!"*60 + "\n")
                rospy.logfatal(f"Motor {max_motor} overheating: {max_temp}°C - DISABLING")
                self.temp_critical_issued = True
                # Auto disable motors
                try:
                    self.dxl_arm.fingers['arm'].deactive_motors(None)
                except:
                    pass
            return True

        # Warning threshold
        elif max_temp >= self.TEMP_WARNING:
            if not self.temp_warning_issued:
                print("\n" + "*"*60)
                print(f"*** WARNING: Motor {max_motor} temperature {max_temp}°C ***")
                print(f"*** Approaching limit ({self.TEMP_CRITICAL}°C) ***")
                print("*** Consider reducing load or taking a break ***")
                print("*"*60 + "\n")
                rospy.logwarn(f"Motor {max_motor} getting hot: {max_temp}°C")
                self.temp_warning_issued = True
        else:
            # Reset warning flags when temperature drops
            if max_temp < self.TEMP_WARNING - 5:
                self.temp_warning_issued = False
            if max_temp < self.TEMP_CRITICAL - 5:
                self.temp_critical_issued = False

        return False

    def simple_throw(self):
        self.switch_mode("current", 13)
        self.switch_mode("current", 14)
        self.up(12,False)
        self.torque_till_position(13, -300, -pi/2)
        self.torque_till_position(14,-300, -0.4)
        with self.lock:
            self.dxl_arm.fingers['arm'].send_single_motor_pos_cmd(2, 15)

        self.rest(4.0)
        self.switch_mode("ext_position", 13)
        self.switch_mode("ext_position", 14)
        print("throw")
        self.press_enter()
    
    def complex_throw(self):

        self.switch_mode("current", 13)
        self.switch_mode("current", 14)

        def arm():
            self.up(12,False)
        
        def elbow():
            self.torque_till_position(13, -500, -pi/2)
        
        def wrist():
            self.torque_till_position(14,-300, 0)

        def hand():
            self.move_when_position(15, 2, 13, 0.2)
      

        arm_thread = threading.Thread(target = arm)
        elbow_thread = threading.Thread(target = elbow)
        wrist_thread = threading.Thread(target = wrist)
        hand_thread = threading.Thread(target = hand)

        hand_thread.start()
        arm_thread.start()
        elbow_thread.start()
        wrist_thread.start()
        
        arm_thread.join()
        elbow_thread.join()
        wrist_thread.join()
        hand_thread.join()


        self.rest(4.0)
        self.switch_mode("ext_position", 13)
        self.switch_mode("ext_position", 14)
        print("throw")
        self.press_enter()

    def simple_squeeze(self, current, time = None):

        with self.lock:
            self.dxl_arm.fingers['arm'].send_motor_torque_cmd(15, current)
        if time:
            self.rest(time)
            with self.lock:
                self.dxl_arm.fingers['arm'].send_motor_torque_cmd(15, 0)


    def dynamic_squeeze(self, start, end, increment, time, rest_time = 1):

        for num_squeeze in range((start-end)//increment):
            with self.lock:
                self.dxl_arm.fingers['arm'].send_motor_torque_cmd(15, start + num_squeeze * increment)
            self.rest(rest_time)
        self.rest(time)
        with self.lock:
                self.dxl_arm.fingers['arm'].send_motor_torque_cmd(15, 0)

    def self_dynamic_squeeze(self, start, max, time, rest_time = 1, increment = 50, i = 0):

        """ Will squeeze with different torques depending
          on the hardness of the object up to a maximum, recursive 
          i is a counter so the current doesn't become too large, DO NOT CHANGE i  """
        print(start)
        if i* abs(increment) > 1000:
            with self.lock:
                self.dxl_arm.fingers['arm'].send_motor_torque_cmd(15, 0)
            print("Infinite recursion")

        elif start < max: # base case, if current already hit maximum (technically minimum)
            with self.lock:
                self.dxl_arm.fingers['arm'].send_motor_torque_cmd(15, 0)
            print("reached max squeeze")
        else:
            moved = False

            with self.lock:
                self.dxl_arm.fingers['arm'].send_motor_torque_cmd(15, start) # start current 
            
            s =  rospy.Time.now().to_sec()
            # start = self.dxl_arm.fingers['arm'].get_motor_pos()[15%11] CAN ALSO USE POSITIONAL DATA TO CHECK IF MOVED. PROBABLY EASIER!
            while (rospy.Time.now().to_sec()-s) < time:
                if self.is_moving(15):
                    moved = True
                self.rate.sleep()

            if moved: # Other base case, if the object gave way at all
                with self.lock:
                    self.dxl_arm.fingers['arm'].send_motor_torque_cmd(15, 0)
                print("moved")
            else: # Recursive case
                self.rest(rest_time)
                self.self_dynamic_squeeze(start - abs(increment), max, time, rest_time, increment, i+1)
    
    def pick_place(self):
        self.pathfinder([self.default,1.5707963267948966,"start", [-0.09510721463796083, 0.86, -0.8513629697430365, 0.8452270204115552, 1.2609375876194162], 
                         -0.15, [-0.13805885995833023, 0.10737911330092352, -0.4402543645337864, 1.7149978381490356, -0.15], "csv","start",
                         [-1.0523153103490506, 0.6, -0.9, 1.8484547361087549, -0.15], 
                         1.5707963267948966, [-1.0032277156971998, 0.1702725939486073, -0.48320600985415585, 1.3483748655930254, 0.6258668318110971],"csv", self.default])
        print("pick_place")
    
    def push(self):
        # self.home()
        # self.press_enter()
        self.pathfinder([self.default,[-1.2,14],[2.99-pi/2,13]])
                          
        """[1.5, 14], [0.4678661365254525, 13],
        [1.08, 12], [1.5, 14]], enter=True)"""

        # push_start = len(self.data)
        self.set_velocity([1,1,2,1])

        def arm():
            with self.lock:
                self.dxl_arm.fingers['arm'].send_single_motor_pos_cmd(1.08, 12)
            
        
        def elbow():
            with self.lock:
                self.dxl_arm.fingers['arm'].send_single_motor_pos_cmd(-pi/2+0.7, 13)

        
        def wrist():
            with self.lock:
                self.dxl_arm.fingers['arm'].send_single_motor_pos_cmd(0, 14)

        arm_thread = threading.Thread(target = arm)
        elbow_thread = threading.Thread(target = elbow)
        wrist_thread = threading.Thread(target = wrist)

        
        elbow_thread.start()
        wrist_thread.start()
        arm_thread.start()


        # elbow_thread.join()
        # wrist_thread.join()
        # arm_thread.join()

                         
        print("push")
        # self.press_enter(csv=push_start-1)

    def baseball(self):
        self.pathfinder([self.default,1.5707963267948966, [0, 0.88, -0.8513629697430365, 0.8452270204115552, 1.2609375876194162], 
                         0.15, [0, -0.9, -2.081620810705046, -1.0691891710106243, 0.15]])
        
        self.set_velocity([20,20,20,20])
        
        self.switch_mode("current", 13)
        self.switch_mode("current", 14)

        def arm():
            self.up(12,False)
        
        def elbow():
            self.torque_till_position(13, 100, -pi/2)
        
        def wrist():
            self.torque_till_position(14, 300, 0)

        def hand():
            self.move_when_position(15, pi/2, 12, -0.2)
      

        arm_thread = threading.Thread(target = arm)
        elbow_thread = threading.Thread(target = elbow)
        wrist_thread = threading.Thread(target = wrist)
        hand_thread = threading.Thread(target = hand)

        hand_thread.start()
        arm_thread.start()
        elbow_thread.start()
        wrist_thread.start()
        
        arm_thread.join()
        elbow_thread.join()
        wrist_thread.join()
        hand_thread.join()


        self.rest(4.0)
        self.switch_mode("ext_position", 13)
        self.switch_mode("ext_position", 14)
        print("Baseball!")
        self.press_enter()

    def random_sample(self, enter = True):
        """Example code usage:
            for i in range (10):
                self.random_sample(enter=False)
                self.rest(5)
                while self.is_moving():
                    self.rate.sleep()    """
        
        data_start = len(self.data)
        pause = enter


        r_11 = np.random.randint(-79, 79)/100
        r_12 = np.random.randint(-61,61)/100
        r_13 = np.random.randint(-78.5, 78.5)/100
        r_14 = np.random.randint(-157, 158)/100
        r_15 = np.random.randint(-90, 158)/100

        length = self.l_12*cos(r_12) - self.l_13*sin(r_12 + r_13) - self.l_grip*(sin(r_12 + r_13 + r_14))

        if length > 0:
            with self.lock:
                self.dxl_arm.fingers['arm'].send_motor_pos_cmd([r_11, r_12, r_13, r_14, r_15]) 
                if enter: 
                    print("enter")
                    self.press_enter()
                self.csv(self.data[data_start-1:], f"arm_data_{self.csv_store}.csv")
                self.csv_store += 1
                
                return True
        else:
            self.random_sample(enter=pause)


    def dynamic_torque(self, torques, frequency, error=0):
        start = rospy.Time.now().to_sec()
        for i in range(len(torques)):
            
            time = rospy.Time.now().to_sec()
                
            with self.lock:
                self.dxl_arm.fingers['arm'].send_torque_sync_cmd(torques[i])
            self.rest(frequency - error)
            # if i <11:
            #     error = rospy.Time.now().to_sec()-time - frequency
            #     print(error)

            # with self.lock:
            #     if self.dxl_arm.fingers['arm'].get_motor_pos()[1] > pi/2:
            #         break
        print(rospy.Time.now().to_sec()-start, "total")
        print("before zero")
        with self.lock:
            self.dxl_arm.fingers['arm'].send_torque_sync_cmd([0,0,0,0,0])
        print("after zero")
        self.press_enter()
    
    def pick_with_position(self):
        
        with self.lock:
            # data_start = len(self.vel)
            
            self.dxl_arm.fingers['arm'].send_motor_pos_cmd([0,0,0,pi/2,0])
            start = rospy.Time.now().to_sec()
        data_start = len(self.vel2)
            
            
            # lentime = rospy.Time.now().to_sec()
            # print(lentime-start)
        self.press_enter()
        with self.lock:
            self.dxl_arm.fingers['arm'].send_motor_pos_cmd([0,0,0,0,0])
            

        self.press_enter()
        self.current_profile = self.vel2[data_start:]
        self.time_profile = self.sub_time[data_start:]
        end = rospy.Time.now().to_sec()
        return self.current_profile, (end-start)
    
    def shake(self, upper_angle = -pi/6, lower_angle = pi/6, duration_time = 20, loop_num = 10):
        arm = self.dxl_arm.fingers['arm']
        self.switch_mode("current",15)
        self.home() # Go to home position, 
        with self.lock:
            arm.send_motor_torque_cmd(15,-100)
        self.press_enter() # Press enter to continue
        start = rospy.Time.now().to_sec()
        data_start = len(self.vel2)
        for i in range(loop_num):
            if i%2 == 0: # If on an even number, go to upper angle, else go to lower angle
                with self.lock:
                    arm.send_single_motor_pos_cmd(upper_angle, 13)
            else:
                with self.lock:
                    arm.send_single_motor_pos_cmd(lower_angle, 13)
            self.rest(duration_time/loop_num) # control frequency = duration_time/loop_num

        self.press_enter()
        self.current_profile = self.vel2[data_start:]
        # self.time_profile = self.sub_time[data_start:]
        end = rospy.Time.now().to_sec()
        return self.current_profile, (end-start)
        
        

        


########################### MEASURE #####################################################################################################
        


    def measure(self):
        # dxl_arm.fingers['arm'].deactive_motors()
        key = None
        poses = []

        while key != "x":
            # i += 1
            key = input("Type 'position' to get positions.\n Type 'torque' to get torques.\n Type 'keep' to stay in present position.\n Type 'deactivate' to deactivate motors.\n Type 'send_pos' to send a motor command\n Type 'x' to exit ")
            if key == "torque":
                print(self.dxl_arm.fingers['arm'].get_motor_torque())
            elif key == "position":
                with self.lock:
                    val = self.dxl_arm.fingers['arm'].get_motor_pos()
                print(val)
                poses.append(val)
            elif key == "open":
                poses.append(pi/2)
            elif key ==  "close":
                poses.append(0)
            elif key == "keep":
                self.switch_mode("ext_position")
                pos = self.dxl_arm.fingers['arm'].get_motor_pos()
                with self.lock:
                    self.dxl_arm.fingers['arm'].send_motor_pos_cmd(pos)
                print(pos)
            elif key == "deactivate":
                self.disable()
            # elif key == "velocity":
            #     print(self.dxl_arm.fingers['arm'].get_motor_vel())
            elif key == "send_pos":
                id = int(input("Input motor id: "))
                move = float(input("Input desired position: "))
                self.switch_mode("ext_position", id)
                with self.lock:
                    self.dxl_arm.fingers['arm'].send_single_motor_pos_cmd(move, id)

            elif key == "x":
                break
            else:
                print("Not recognized")


        self.switch_mode("ext_position")
        return poses
    
    def pathfinder(self, path, enter=True, rest=None,csv=None):
        
        self.data_start = len(self.data)
        
        for i in range(len(path)):
            csv = False
            if isinstance(path[i], float):
                with self.lock:
                    self.dxl_arm.fingers['arm'].send_single_motor_pos_cmd(path[i], 15)
            elif len(path[i]) == 2:
                
                with self.lock:
                    self.dxl_arm.fingers['arm'].send_single_motor_pos_cmd(path[i][0],path[i][1])      
            elif path[i] == "csv":
                csv = True    
            elif path[i] == "start":
                self.data_start = len(self.data)
            else:
                self.data_start = len(self.data) 
                with self.lock:
                    self.dxl_arm.fingers['arm'].send_motor_pos_cmd(path[i])
            if enter:
                if csv:
                    self.press_enter(self.data_start-1)
                else:
                    self.press_enter()
            elif rest:
                self.rest(rest)
            else:
                self.rate.sleep()
                continue
            
    
    def is_moving(self, id=None, precision = 0.1):
        if id:
            with self.lock:
                vel = self.dxl_arm.fingers['arm'].motor_status.motorsVel[id%11]
            # print(vel)
            if vel < precision:
                return False
            else:
                return True
        else:
            for d in range(11,16):
                bol = self.is_moving(id=d)
                if bol == True:
                    return True
            return False
        
        
    def torque_motion(self, function, frequency):
        """Current reproduce function"""
        # self.live_current = 1
        arm = self.dxl_arm.fingers['arm']
        # print("before home")
        with self.lock:
            arm.send_motor_pos_cmd([0.036815695988888064, 0.15800069528564462, 0.40190468121202805, 0.9848198677027558, 0.004601961998611008])
            print("after home")
        self.press_enter()
        self.rate = rospy.Rate(frequency)
        t_profile, time = function()
        # print(t_profile)
        print(time, "time")
        print(len(t_profile), "length")
        

        with self.lock:
            arm.send_motor_pos_cmd([0.036815695988888064, 0.15800069528564462, 0.40190468121202805, 0.9848198677027558, 0.004601961998611008])
        # print("after home")
        self.press_enter()
        self.switch_mode("current")
        self.rate = rospy.Rate(frequency)
        self.press_enter()
        
        self.dynamic_torque(t_profile,1/frequency)
        # self.live_current = 0

    def force_meter(self, complex = False):
        
        arm = self.dxl_arm.fingers['arm']
        # print(start)


        ## HOME ##
        with self.lock:
            arm.send_motor_pos_cmd([0.036815695988888064, 0.15800069528564462, 0.40190468121202805, 0.9848198677027558, pi/2])
            # arm.send_motor_pos_cmd([0.036815695988888064, 0.23, 0.40190468121202805, 0.9848198677027558, pi/2])
            # arm.send_motor_pos_cmd([0.036815695988888064, 0.16, 0.2, pi/2.5, pi/2])
        print("home")
        self.press_enter()
        ## SQUEEZE ##
        self.switch_mode("current", 15)
        with self.lock:
            arm.send_motor_torque_cmd(15,-150)
        print("squeeze")
        self.press_enter()

        ## UP ##
        start = len(self.data)
        with self.lock:
            self.dxl_arm.fingers['arm'].send_motor_pos_cmd([0,0,-pi/6,pi/4,0])
            # self.dxl_arm.fingers['arm'].send_motor_pos_cmd([0,-pi/4,0,pi/4,0])
        self.rest(6)
        if complex == True:
            self.pathfinder([[0.6887603124587809,0 , 0.3, 0.8605668937402585, -0.9449361970481269],
                             [-0.2377680365949021,0.2515739225907351 , -0.3282732892342519, -1.1827042336430291, -0.8498289824101661],
                              [-0.06596145531342446,-0.5675753131620244 , -1.3529768275916363, -0.9786839183712743, -0.8513629697430365],
                               [-0.09664120197083118,-1.41587030823932 , -0.012271898662962688, 1.543191256867558, -0.8498289824101661]],
                               enter = False, rest = 6)
        # print("home")
            self.rest(2)
        ## HOME ##
        with self.lock:
            arm.send_motor_pos_cmd([0.036815695988888064, 0.15800069528564462, 0.40190468121202805, 0.9848198677027558, pi/2])
        print("home_2")
        self.rest(5)
        self.disable()
        self.press_enter(csv=start)

    def weightlifting(self, weights, complexity=False):
        """Example code:
            weights = [0,5,10,20,100]
            self.weightlifting(weights, complexity=False)"""

        for i in range(len(weights)):
            self.switch_mode("ext_position")
            self.set_velocity([0.5,0.5,0.5,0.5,1])
            self.weights = [0,0, weights[i]/1000]
            self.force_meter(complex=complexity)

    # def complex_push(self):
    def force_pull(self):
        """Example code:
            for _ in range(3):
                self.force_pull()"""
        
        arm = self.dxl_arm.fingers['arm']
        self.disable()
        input("Set position, then press enter to keep")
        self.switch_mode("ext_position")
        self.set_velocity([0.5,0.5,0.5,0.5,1])
        with self.lock:
            new_pos = arm.get_motor_pos()
        with self.lock:
            arm.send_motor_pos_cmd(new_pos)
        self.press_enter()
        start = len(self.data)
        with self.lock:
            arm.send_single_motor_pos_cmd(-0.1,11)
        self.rest(5)
        with self.lock:
            arm.send_motor_pos_cmd(new_pos)
        self.press_enter(csv=start)
    
    def force_push(self,high=False,mid=False,low=False):
        self.force_vector = [1,0,0]
        arm = self.dxl_arm.fingers['arm']
        if high is True:
            with self.lock:
                arm.send_motor_pos_cmd([-0.01533987332870336, -0.2899236059124935, -0.38042885855184333, 0.6, -0.9341982857180346])
            # self.press_enter() # high
            # self.switch_mode("current", 15)
            # with self.lock:
            #     arm.send_motor_torque_cmd(15,-150)
            self.press_enter() # high
            start = len(self.data)
            with self.lock:
                arm.send_motor_pos_cmd([-0.009203923997222016, 0.21782620126758773, -0.9, 0.6, -0.9341982857180346]) # push
            self.rest(5)
            with self.lock:
                arm.send_motor_pos_cmd([-0.01533987332870336, -0.2899236059124935, -0.38042885855184333, 0.55, -0.9341982857180346]) # high
            self.press_enter(csv=start)

            
        elif mid is True:
            with self.lock:
                arm.send_motor_pos_cmd([0.009203923997222016, -0.5108177818458219, 1.0599852470134021, -0.4740020858569338, -0.8682368304046102])

            # self.press_enter() # high
            # self.switch_mode("current", 15)
            # with self.lock:
            #     arm.send_motor_torque_cmd(15,-150)
            self.press_enter()
            start = len(self.data)
            with self.lock:
                arm.send_motor_pos_cmd([-0.00766993666435168, -0.07516537931064646, 0.7010322111217435, -0.579847211824987, -0.8682368304046102]) # push
            self.rest(5)
            with self.lock:
                arm.send_motor_pos_cmd([0.009203923997222016, -0.5108177818458219, 1.0599852470134021, -0.4740020858569338, -0.8682368304046102]) # high
            self.press_enter(csv=start)

        elif low is True:
            with self.lock:
                arm.send_motor_pos_cmd([-0.03067974665740672, 0.41571056720786104, 1.0599852470134021, -1.4526860042282081, -0.8651688557388695])
            self.press_enter() # high
            start = len(self.data)
            with self.lock:
                arm.send_motor_pos_cmd([-0.013805885995833024, 0.4632641745268415, 0.6795563884615589, -1.1090728416652529, -0.8697708177374806]) # push
            self.rest(5)
            with self.lock:
                arm.send_motor_pos_cmd([-0.03067974665740672, 0.41571056720786104, 1.0599852470134021, -1.4526860042282081, -0.8651688557388695]) # high
            self.press_enter(csv=start)

        
        else:
            rospy.logwarn("This did nothing")

        # self.force_vector = [0,0,1]    
        

##################################################################        


    def run_sequence(self):
        ### ALREADY STARTS IN EXTENDED POSITION MODE ###
        arm = self.dxl_arm.fingers['arm']
        self.rest(2)
        self.set_velocity([0.5,0.5,0.5,0.5,1])#rad/s

        if self.twin:
            self.terminal_thread = threading.Thread(target=self.press_enter)
            self.terminal_thread.start()
            while self.terminal_thread.is_alive():
                # Read leader position (use leader's lock)
                with self.twin.lock:
                    raw_pos = self.twin.get_pos()
                if raw_pos is False or raw_pos is None:
                    # Communication failed, skip this iteration
                    self.rate.sleep()
                    continue
                leader_pos = list(raw_pos)
                leader_pos[4] = leader_pos[4] + self.gripper_follow_offset
                # Store goal position for CSV recording
                self.goal_pos = leader_pos.copy()
                # Send to follower (use follower's lock)
                with self.lock:
                    arm.send_motor_pos_cmd(leader_pos)
                self.rate.sleep()
        else:
            # INSERT MANUAL FUNCTIONS HERE
            pass

        
        


##################################################################    
        
    def get_width(self, angle,mujoco=True):
        if mujoco:
            c = 0.019 
            k = pi/0.029
            self.ap_theta = sqrt(self.ap_length**2 - (self.ap_radius**2)*(sin(k*(angle-c))**2) + self.ap_radius*cos(k*(angle-c)))

            return 2 * self.ap_theta - 2 * self.ap_delta
        else:
            c = -1.237927777626361 # minimum angle of gripper in radians
            k = pi/(pi-c)
            self.ap_theta = sqrt(self.ap_length**2 - (self.ap_radius**2)*(sin(k*(angle-pi))**2) + self.ap_radius*cos(k*(angle-pi)))

            return 2 * self.ap_theta - 2 * self.ap_delta
    

    def record_data(self):
        """If you want to plot velocity, uncomment the relevant lines"""
        # Set PID gains for Extended Position Control (Mode 4) (from PID_CONFIG)
        print("Setting PID gains from PID_CONFIG:")
        for motor_id, gains in PID_CONFIG.items():
            self.set_single_motor_pid_gains(
                motor_id=motor_id,
                kp=gains['kp'],
                ki=gains['ki'],
                kd=gains['kd']
            )
            print(f"  Motor {motor_id}: kp={gains['kp']}, ki={gains['ki']}, kd={gains['kd']}")
        # Verify PID gains
        print("Verifying PID gains:")
        self.get_pid_gains()

        self.data = []
        self.vel2 = []
        self.sub_time = []
        self.weights = []
        self.force_count = 0
        self.force_reading = None
        cache = self.force_count

        # In twin mode: wait for user to press Enter before starting recording
        # This allows follower to track leader without recording data
        if self.twin:
            print("\n" + "="*60)
            print("  FOLLOWER IS READY - Following leader movements")
            print("  Press ENTER to start recording CSV/video...")
            print("="*60)
            input()  # Block until Enter
            print(">>> Recording started!")

        # Record the actual start time (for CSV timestamp and video sync)
        import time as time_module_local
        self.recording_start_time = time_module_local.time()

        # Signal that recording has started (for video synchronization)
        self.recording_started_event.set()

        if self.run_thread.is_alive():
            start = rospy.Time.now().to_sec()
        # count = 0
        while self.run_thread.is_alive():
            
            
            # count += 1
            # print(self.live_current, "current")
            while self.live_current == 1:
                self.rate.sleep()
            

            try:
                with self.lock:
                    self.dxl_arm.update_motor_status_sync()

            except:
                rospy.logwarn("update motor status failed")

            arm_status = MotorMonitorNoLength()
            # print(self.dxl_arm.fingers['arm'].motor_status.motorsPos, "pos")
            arm_status.motorsPos = self.dxl_arm.fingers['arm'].motor_status.motorsPos
            arm_status.current = self.dxl_arm.fingers['arm'].motor_status.torque
            # arm_status.stamp = (rospy.Time.now().to_sec() - start)
            arm_status.relative_time = (rospy.Time.now().to_sec() - start)
            arm_status.motorsVel = self.dxl_arm.fingers['arm'].motor_status.motorsVel
            arm_status.coilsTemp = self.dxl_arm.fingers['arm'].motor_status.coilsTemp
            arm_status.pwm = self.dxl_arm.fingers['arm'].motor_status.pwm
            arm_status.motorsVolts = self.dxl_arm.fingers['arm'].motor_status.motorsVolts

            # Check motor temperatures
            if self.check_temperature(list(arm_status.coilsTemp)):
                rospy.logfatal("Motor overheating! Stopping data recording.")
                break  # Exit the recording loop

            # Handle case where gripper motor (index 4) is not responding
            if len(arm_status.motorsPos) > 4:
                arm_status.aperture = self.get_width(arm_status.motorsPos[4])
            else:
                arm_status.aperture = 0.0  # Default value when gripper not available

            if self.force_count == cache or self.force_reading == None:

                force_direction = [-999]*3
            else:
                force_direction = [float(self.force_reading) * vec for vec in self.force_vector]
                cache = self.force_count
            arm_status.force = force_direction

            self.arm_status = arm_status # global for get_data

            # Get force data from 6-axis force sensor (if available)
            # Updates force_x, force_y, force_z columns (not force_gripper columns)
            if self.force_sensor is not None:
                force_data = self.get_force_from_sensor()
            else:
                force_data = arm_status.force  # Use force gauge data or default [-999, -999, -999]

            # Get gripper force (from weights if provided, otherwise default)
            if self.weights:
                gripper_force = self.weights
            else:
                gripper_force = [0.0, 0.0, 0.0]

            self.data.append([
                arm_status.relative_time,
                *arm_status.motorsPos,
                arm_status.aperture,
                *self.goal_pos,  # Goal position from leader arm
                self.get_width(self.goal_pos[4]),
                *arm_status.current,
                *arm_status.motorsVel,
                *arm_status.coilsTemp,
                *arm_status.pwm,
                *arm_status.motorsVolts,
                *force_data,
                *gripper_force
            ])
            
            self.vel2.append([arm_status.current])
            self.sub_time.append([arm_status.relative_time])
            self.rate.sleep()
        # print(self.vel)
            
    def get_data(self):
        return self.arm_status
    
    
    def record_force_data(self):
        if self.force_gauge is not None:
            while self.run_thread.is_alive():
                self.force_reading = self.force_gauge.main(once = True)
                self.force_count += 1
                # print(self.force_reading)
                self.rate.sleep()
        else:
            return None

    def record_force_sensor_data(self):
        """Continuously read 6-axis force sensor data in a separate thread."""
        if self.force_sensor is None or not self.force_sensor.connected:
            return

        while self.run_thread.is_alive():
            # Read latest data (drain buffer to avoid lag)
            raw_data = self.force_sensor.read_latest()
            if raw_data:
                # Apply tare offset and convert to SI units
                data = self.force_sensor.apply_offset(raw_data)
                with self.force_sensor_lock:
                    self.force_sensor_data = {
                        'fx': data['fx'],
                        'fy': data['fy'],
                        'fz': data['fz']
                    }
            time_module.sleep(0.005)  # ~200Hz sampling

    def get_force_from_sensor(self):
        """
        Get force from 6-axis force sensor for force_x, force_y, force_z columns.

        Coordinate mapping:
            - force_x (CSV) = -sensor_y
            - force_y (CSV) = -sensor_x
            - force_z (CSV) = -sensor_z

        Returns:
            [force_x, force_y, force_z] in Newtons
        """
        if self.force_sensor is None:
            return [-999.0, -999.0, -999.0]

        with self.force_sensor_lock:
            fx = self.force_sensor_data['fx']
            fy = self.force_sensor_data['fy']
            fz = self.force_sensor_data['fz']

        # Apply coordinate transformation: sensor -> CSV columns
        # force_x = -sensor_y, force_y = -sensor_x, force_z = -sensor_z
        return [-fy, -fx, -fz]

    def run(self):
        self.run_thread = threading.Thread(target = self.run_sequence)
        self.data_thread = threading.Thread(target = self.record_data)
    

        self.run_thread.start()
        self.data_thread.start()

        self.data_thread.join()

    def run_with_force(self):
        self.run_thread = threading.Thread(target = self.run_sequence)
        self.data_thread = threading.Thread(target = self.record_data)
        self.force_thread = threading.Thread(target = self.record_force_data)

        self.run_thread.start()
        self.data_thread.start()
        self.force_thread.start()

        self.data_thread.join()

    def run_with_force_sensor(self):
        """Run data collection with 6-axis force sensor."""
        self.run_thread = threading.Thread(target=self.run_sequence)
        self.data_thread = threading.Thread(target=self.record_data)
        self.force_sensor_thread = threading.Thread(target=self.record_force_sensor_data)

        self.run_thread.start()
        self.data_thread.start()
        self.force_sensor_thread.start()

        self.data_thread.join()

    def csv(self, data=None, name=None):
        if data == None:
            data = self.data
        if name == None:
            name = "arm_data.csv"

        with open(name, "w", newline="") as f:
            write = csv.writer(f)
            write.writerow(["timestamp", "pos1", "pos2", "pos3", "pos4", "pos5", "aperture",
                            "goal_pos1", "goal_pos2", "goal_pos3", "goal_pos4", "goal_pos5", "goal_aperture",
                            "current1", "current2", "current3", "current4", "current5",
                            "vel1","vel2","vel3","vel4","vel5",
                            "temp1","temp2","temp3","temp4","temp5",
                            "pwm1","pwm2","pwm3","pwm4","pwm5",
                            "volts1","volts2","volts3","volts4","volts5",
                            "force_x","force_y","force_z",
                            "force_gripper_x","force_gripper_y","force_gripper_z"
                            ])
            write.writerows(data)


    def plot_current(self, id = None, torque = False):
        # print(self.data)
        data_array = np.array(self.current_profile)
        time = np.array(self.time_profile)[:,0]
        t1,t2,t3,t4,t5 = data_array[:,0], data_array[:,1], data_array[:,2], data_array[:,3], data_array[:,4] 
        motors = {11:t1, 12:t2, 13:t3, 14:t4, 15:t5}

        if id:
            plt.plot(time, motors[id])
            plt.xlabel("Time (s)")
            plt.ylabel("Current (mA)")
        else:
            plt.plot(time, t1, "r-", label="motor 11")
            plt.plot(time, t2, "b-", label="motor 12")
            plt.plot(time, t3, "y-", label="motor 13")
            plt.plot(time, t4, "k-", label="motor 14")
            plt.plot(time, t5, "g-", label="motor 15")
            
        plt.show() 
    



# RUNS THE CODE

def main():
    
    dxl = DXL_ARM()
    
    # dxl.measure()
    
    dxl.run()

    dxl.csv()

    dxl.disable()

    # dxl.plot_current()

def force_gauge_run():
    sensor = ForceGaugeReader(timeout = 0.1)
    dxl = DXL_ARM(siblings=False, twin=None, force_gauge=sensor)

    dxl.run_with_force()

    dxl.csv()

    dxl.disable()


def get_trajectory_filepath(with_video=True, force_sensor_direction=None):
    """
    Interactive menu to select trajectory category and generate filepath.
    Returns tuple of (csv_filepath, video_filepath) if with_video=True,
    otherwise returns just csv_filepath.

    Args:
        with_video: If True, also return video filepath
        force_sensor_direction: If provided ('x', 'y', or 'z'), auto-select force_labeled/force_sensor category

    Directory structure:
    dataset/
    ├── trajectory_data/
    │   ├── force_labeled/
    │   │   ├── weight/
    │   │   │   ├── static_holding/
    │   │   │   ├── slow_motion_load/
    │   │   │   └── pick_place_object/
    │   │   └── force_sensor/
    │   │       ├── force_x/
    │   │       ├── force_y/
    │   │       └── force_z/
    │   └── force_unlabeled/
    │       ├── s_curve/
    │       ├── circular_cw/
    │       ├── circular_ccw/
    │       ├── backward_forward/
    │       ├── joint_sweep/
    │       ├── pick_place_empty/
    │       └── go_up_and_stay_still/
    └── video_data/
        └── (same structure as trajectory_data)
    """
    import os

    # Base directories for data
    DATASET_DIR = "dataset"
    BASE_DIR = os.path.join(DATASET_DIR, "trajectory_data")
    VIDEO_DIR = os.path.join(DATASET_DIR, "video_data")

    # Category definitions: key -> subdirectory path
    CATEGORIES = {
        # Force-labeled / Weight
        "1": "force_labeled/weight/go_up_and_stay_still_with_object",
        "2": "force_labeled/weight/pick_place_object",

        # Force-labeled / Force sensor
        "3": "force_labeled/force_sensor/force_x",
        "4": "force_labeled/force_sensor/force_y",
        "5": "force_labeled/force_sensor/force_z",

        # Force-unlabeled
        "6": "force_unlabeled/s_curve",
        "7": "force_unlabeled/circular_cw",
        "8": "force_unlabeled/circular_ccw",
        "9": "force_unlabeled/backward_forward",
        "10": "force_unlabeled/joint_sweep",
        "11": "force_unlabeled/pick_place_empty",
        "12": "force_unlabeled/go_up_and_stay_still",
    }

    # If force sensor direction is provided, auto-select corresponding category
    if force_sensor_direction:
        direction_map = {'x': '3', 'y': '4', 'z': '5'}
        choice = direction_map.get(force_sensor_direction.lower(), '3')
        print(f"\n>>> Auto-selected: force_labeled/force_sensor/force_{force_sensor_direction}")
    else:
        print("\n" + "="*60)
        print("  SELECT TRAJECTORY CATEGORY")
        print("="*60)
        print("\n[Force-labeled / Weight]")
        print("  1. Go up and stay still (with object)")
        print("  2. Pick & place (with object)")
        print("\n[Force-labeled / Force sensor]")
        print("  3. Force X direction")
        print("  4. Force Y direction")
        print("  5. Force Z direction")
        print("\n[Force-unlabeled]")
        print("  6. S-curve around obstacles")
        print("  7. Circular trajectory (CW)")
        print("  8. Circular trajectory (CCW)")
        print("  9. Backward lean + forward reach")
        print("  10. Joint sweep (full range)")
        print("  11. Pick & place (empty grasp)")
        print("  12. Go up and stay still")
        print("\n  0. Custom path")
        print("="*60)

        choice = input("Enter category number: ").strip()

    if choice == "0":
        # Custom path
        custom_path = input("Enter custom filepath (with .csv): ").strip()
        if not custom_path.endswith('.csv'):
            custom_path += ".csv"
        if with_video:
            # Generate video path based on custom CSV path
            video_path = custom_path.replace('.csv', '.mp4')
            # If path is in trajectory_data, put video in video_data
            if 'trajectory_data' in video_path:
                video_path = video_path.replace('trajectory_data', 'video_data')
            print(f">>> Video will save to: {video_path}")
            return custom_path, video_path, choice
        return custom_path, choice

    if choice not in CATEGORIES:
        print(f"Invalid choice '{choice}', using default category 6 (s_curve)")
        choice = "6"

    subdir = CATEGORIES[choice]
    full_dir = os.path.join(BASE_DIR, subdir)

    # Create directory if not exists
    if not os.path.exists(full_dir):
        os.makedirs(full_dir)
        print(f"Created new directory: {full_dir}")
    else:
        print(f"Using existing directory: {full_dir}")

    # Find next available number (files like 001.csv, 002.csv, etc.)
    existing_files = [f for f in os.listdir(full_dir) if f.endswith('.csv')]
    print(f"Found {len(existing_files)} existing files in this category")

    existing_nums = []
    for f in existing_files:
        try:
            num_str = f[:-4]  # Remove .csv
            existing_nums.append(int(num_str))
        except:
            pass

    next_num = max(existing_nums, default=0) + 1

    # Ask for custom name or use number
    print(f"\nNext auto number: {next_num:03d}")
    custom_name = input(f"Enter filename (or press Enter for '{next_num:03d}'): ").strip()

    if custom_name:
        if not custom_name.endswith('.csv'):
            custom_name += ".csv"
        filename = custom_name
    else:
        filename = f"{next_num:03d}.csv"

    csv_filepath = os.path.join(full_dir, filename)
    print(f"\n>>> CSV will save to: {csv_filepath}")

    if with_video:
        # Create corresponding video path
        video_subdir = os.path.join(VIDEO_DIR, subdir)
        if not os.path.exists(video_subdir):
            os.makedirs(video_subdir)
            print(f"Created video directory: {video_subdir}")

        # Replace .csv with .mp4 for video filename
        video_filename = filename.replace('.csv', '.mp4')
        video_filepath = os.path.join(video_subdir, video_filename)
        print(f">>> Video will save to: {video_filepath}")

        return csv_filepath, video_filepath, choice
    else:
        return csv_filepath, choice


def twin_motion(record_video=True, video_resolution="qhd"):
    """
    Twin motion data collection with synchronized video recording.

    Args:
        record_video: If True, record video synchronized with CSV data
        video_resolution: Video resolution preset or "WxH"
            Presets: "4k" (3840x2160), "qhd" (2560x1440), "1080p" (1920x1080)
            Default: "qhd" (2560x1440) - good balance between quality and compatibility

    Video is automatically aligned to CSV (trimmed so video t=0 matches CSV row 0).

    When category 3/4/5 (Force sensor) is selected, automatically connects to
    6-axis force sensor and records force data.
    """
    import time as time_module

    # Interactive category selection and filename generation
    csv_filepath, video_filepath, choice = get_trajectory_filepath(with_video=True)

    # Check if force sensor category is selected (3=x, 4=y, 5=z)
    force_sensor = None
    force_sensor_direction = None
    if choice in ['3', '4', '5']:
        direction_map = {'3': 'x', '4': 'y', '5': 'z'}
        force_sensor_direction = direction_map[choice]

        print("\n" + "="*60)
        print("  6-AXIS FORCE SENSOR MODE")
        print("="*60)

        force_sensor = ForceSensorReader(port="/dev/ttyACM0", baudrate=115200)
        if not force_sensor.connect():
            print("[Error] Could not connect to force sensor!")
            print("Check: Is /dev/ttyACM0 available? Is the sensor powered on?")
            return

        print(f">>> Force direction: {force_sensor_direction.upper()}")

        # Tare (zero) the sensor
        tare_input = input("Tare the sensor? (y/n, default=y): ").strip().lower()
        if tare_input != 'n':
            print("\n>>> Keep the sensor UNLOADED during tare!")
            input("Press Enter when ready...")
            force_sensor.tare(duration=5.0)
        else:
            print(">>> Skipping tare, using raw values")

    older = DXL_ARM(siblings=True, twin=None)
    older.disable()
    younger = DXL_ARM(
        siblings=True,
        twin=older,
        force_sensor=force_sensor,
        force_sensor_direction=force_sensor_direction
    )

    # Initialize video recorder (but don't start yet - wait for user Enter)
    video_recorder = None
    video_start_time_holder = [None]  # Use list to allow mutation in thread

    if record_video:
        video_recorder = VideoRecorder(
            device="/dev/video1",
            resolution=video_resolution,
            fps=30
        )

    def start_video_when_ready():
        """Wait for recording_started_event, then start video."""
        younger.recording_started_event.wait()
        if video_recorder is not None:
            if video_recorder.start(video_filepath):
                video_start_time_holder[0] = time_module.time()
                time_module.sleep(0.3)  # Give ffmpeg a moment to initialize
            else:
                print("[Warning] Video recording failed to start")

    # Start video thread that waits for user Enter
    if record_video and video_recorder is not None:
        video_thread = threading.Thread(target=start_video_when_ready, daemon=True)
        video_thread.start()

    try:
        # Use force sensor mode if sensor is connected
        # Note: run() will block in record_data() until user presses Enter
        if force_sensor is not None:
            younger.run_with_force_sensor()
        else:
            younger.run()

        # Get the actual start times from record_data
        csv_start_time = getattr(younger, 'recording_start_time', time_module.time())
        video_start_time = video_start_time_holder[0]

        younger.csv(name=csv_filepath)
        print(f"\n>>> CSV data saved to: {csv_filepath}")

    finally:
        # Stop video recording AFTER data collection (in finally to ensure cleanup)
        if video_recorder is not None:
            video_recorder.stop()

            # Calculate offset and align video
            video_start_time = video_start_time_holder[0]
            if video_start_time is not None:
                video_offset = csv_start_time - video_start_time
                print(f">>> Aligning video (trimming {video_offset:.3f}s from start)...")

                # Trim video to align with CSV, overwrite original
                temp_path = video_filepath.replace('.mp4', '_temp.mp4')
                cmd = [
                    "ffmpeg", "-y",
                    "-ss", f"{video_offset:.6f}",  # Seek to offset (trim beginning)
                    "-i", video_filepath,
                    "-c", "copy",  # Copy streams without re-encoding (fast)
                    "-avoid_negative_ts", "make_zero",
                    temp_path
                ]
                try:
                    result = subprocess.run(cmd, stdout=subprocess.DEVNULL,
                                          stderr=subprocess.PIPE, timeout=60)
                    if result.returncode == 0:
                        # Replace original with aligned version
                        os.remove(video_filepath)
                        os.rename(temp_path, video_filepath)
                        print(f">>> Video saved (aligned): {video_filepath}")
                    else:
                        print(f"[Warning] Video alignment failed, keeping original")
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                except Exception as e:
                    print(f"[Warning] Video alignment error: {e}")
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
            else:
                print(f">>> Video saved: {video_filepath}")

        # Disconnect force sensor if connected
        if force_sensor is not None:
            force_sensor.disconnect()

        younger.disable()


def publish(disable_motors=True, device=None):
    """
    Publish motor data to ROS topic.

    Args:
        disable_motors: If True, disable motors (free movement, no current).
                       If False, keep motors enabled (will read current/PWM).
        device: USB device path (e.g., '/dev/ttyUSB0' or '/dev/ttyUSB1').
                If None, uses default '/dev/ttyUSB0'.
    """
    arm = DXL_ARM(device=device)
    if disable_motors:
        arm.disable()
    # else: motors stay enabled in ext_position mode, will read current

    arm.run_thread = threading.Thread(target=arm.record_data, daemon=True)
    arm.run_thread.start()

    rate = rospy.Rate(100)

    pub = rospy.Publisher("/dxl_arm/monitor", MotorMonitorNoLength, queue_size=9)

    while not rospy.is_shutdown():
        msg = arm.arm_status
        if msg is not None:
            pub.publish(msg)
        rate.sleep()


def twin_motion_publish():
    """
    Twin motion with real-time ROS publishing (for online inference).

    - Leader arm (ttyUSB0): Disabled, can be moved freely by hand
    - Follower arm (ttyUSB1): Follows leader, publishes sensor data to ROS

    The follower's data is published to /dxl_arm/monitor for real-time simulation.
    NO CSV saving - use original twin_motion() if you need to save data to CSV.
    """
    print("=== Twin Motion Publish Mode (Real-time) ===")
    print("Leader arm (ttyUSB0): Disabled - move by hand")
    print("Follower arm (ttyUSB1): Following leader, publishing to /dxl_arm/monitor")
    print("NOTE: No CSV saving. Use --twin-csv for CSV recording.")

    # Leader arm - disabled, can be moved by hand
    leader = DXL_ARM(siblings=True, twin=None)
    leader.disable()
    print("Leader arm disabled - ready for manual control")

    # Follower arm - will follow leader (twin=leader)
    follower = DXL_ARM(siblings=True, twin=leader)
    print("Follower arm enabled - will follow leader")

    # Set follower velocity profile (same as original twin_motion)
    # Values are in rad/s: [motor11, motor12, motor13, motor14, motor15]
    # Lower values = slower, smoother movement
    follower.set_velocity([0.5, 0.5, 0.5, 0.5, 1.0])  # All 5 motors
    follower.rest(2)  # Wait for velocity setting to take effect

    # ROS publisher
    rate = rospy.Rate(62.5)  # Match simulation frequency (data_dt=0.016s)
    pub = rospy.Publisher("/dxl_arm/monitor", MotorMonitorNoLength, queue_size=9)

    # Start a dummy run_thread so record_data can check is_alive()
    # This thread just waits until rospy is shutdown
    def keep_alive():
        while not rospy.is_shutdown():
            rate.sleep()

    follower.run_thread = threading.Thread(target=keep_alive, daemon=True)
    follower.run_thread.start()

    # Start data recording thread (for arm_status to be updated)
    follower.data_thread = threading.Thread(target=follower.record_data, daemon=True)
    follower.data_thread.start()

    print("Starting twin motion... Press Ctrl+C to stop")
    try:
        while not rospy.is_shutdown():
            # Get leader position and apply gripper offset for calibration
            with leader.lock:
                leader_pos = list(leader.get_pos())
            leader_pos[4] = leader_pos[4] + follower.gripper_follow_offset

            # Send to follower (all 5 motors)
            with follower.lock:
                follower.dxl_arm.fingers['arm'].send_motor_pos_cmd(leader_pos)

            # Publish follower's sensor data to ROS
            msg = follower.arm_status
            if msg is not None:
                pub.publish(msg)

            rate.sleep()
    except KeyboardInterrupt:
        print("\nStopping twin motion publish...")
    finally:
        follower.disable()
    


if __name__ == '__main__':
    import sys

    # Usage:
    #   python dxl_arm_class.py                         # Single arm, motors disabled (USB0)
    #   python dxl_arm_class.py --enabled               # Single arm, motors enabled (USB0)
    #   python dxl_arm_class.py --usb1                  # Single arm on USB1 (follower arm)
    #   python dxl_arm_class.py --usb1 --enabled        # Single arm on USB1, motors enabled
    #   python dxl_arm_class.py --twin-csv              # Twin motion: CSV + VIDEO (default: qhd)
    #   python dxl_arm_class.py --twin-csv --no-video   # Twin motion WITHOUT video
    #   python dxl_arm_class.py --twin-csv --res=1080p  # Twin motion with 1080p video
    #   python dxl_arm_class.py --twin-publish          # Twin motion: publish to ROS
    #
    # USB port selection:
    #   Default: /dev/ttyUSB0 (leader arm)
    #   --usb1:  /dev/ttyUSB1 (follower arm)
    #
    # Video resolution options (--res=XXX):
    #   qhd    -> 2560x1440 (default) [RECOMMENDED]
    #   4k     -> 3840x2160 (highest quality)
    #   1080p  -> 1920x1080 (most compatible)
    #   720p   -> 1280x720  (fastest)
    #
    # Data categories:
    #   force_labeled/weight/        - weight labeled (categories 1-2)
    #   force_labeled/force_sensor/  - 6-axis force sensor labeled (categories 3-5, auto-connect sensor)
    #   force_unlabeled/             - unlabeled (categories 6-12)

    # Parse resolution argument
    video_resolution = "qhd"  # Default: 2560x1440
    for arg in sys.argv:
        if arg.startswith('--res='):
            video_resolution = arg.split('=')[1]
            break

    if '--twin-csv' in sys.argv:
        # Twin motion - saves to CSV (with video by default, auto-aligned)
        # Categories 3-5 auto-connect to 6-axis force sensor
        record_video = '--no-video' not in sys.argv
        twin_motion(record_video=record_video, video_resolution=video_resolution)
    elif '--twin-publish' in sys.argv:
        # Twin motion - publishes to ROS for real-time simulation
        twin_motion_publish()
    else:
        # Single arm mode
        disable = '--enabled' not in sys.argv
        device = '/dev/ttyUSB1' if '--usb1' in sys.argv else None
        publish(disable_motors=disable, device=device)
    