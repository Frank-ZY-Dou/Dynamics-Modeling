#!/usr/bin/env python

import rospy
from soft_hand_control.msg import MotorPosCmd, MotorMonitor, MotorPosTraj, FingerMeasure

from math import pi
import dynamixel_sdk
import numpy as np


class FingerControlDXL:
    """Create a finger control object
    """

    def __init__(self, finger_name, motor_ids, motor_dirs, motor_limits, DXL_PORT,
                 forward_model, backward_model, forward_model_const=None, backward_model_const=None, home_offset=None, group_read=False):
        """Initialize a finger controller
        motor_ids -- [motor_0_id, motor_1_id]
        motor_dirs -- 0 or 1
        motor_limits -- max and min position for each motor
        forward_model -- 3x2 numpy array
        backward_model -- 3x2 numpy array
        forward_model_const -- 3x1 numpy array, required by task space controller
        backward_model_const -- 3x1 numpy array, required by task space controller
        finger_name -- name of the finger, e.g. index, middle, ring, little, thumb
        """
        self.motor_ids = motor_ids
        self.__DXL_PORT = DXL_PORT
        self.__DXL_IDs = motor_ids
        self.__DXL_DIRs = motor_dirs
        self.__DXL_PROTOCOL_VERSION = 2.0
        self.__DXL_BAUDRATE = 1000000
        self.__DXL_ADDR_DRIVE_MODE = 10
        self.__DXL_ADDR_OPERATING_MODE = 11
        self.__DXL_ADDR_TORQUE_ENABLE = 64
        self.__DXL_ADDR_GOAL_POS = 116
        self.__DXL_ADDR_PRESENT_POS = 132
        self.__DXL_ADDR_MIN_POS = 52
        self.__DXL_ADDR_MAX_POS = 48
        self.__DXL_ADDR_HOME_OFFSET = 20
        self.__DXL_ADDR_GOAL_VEL = 104
        self.__DXL_ADDR_PRESENT_VEL = 128
        self.__DXL_ADDR_GOAL_CURRENT = 102
        self.__DXL_ADDR_PRESENT_CURRENT = 126
        self.__DXL_ADDR_PRESENT_TEMP = 146
        self.__DXL_ADDR_INDIRECTADDR_READ = 168
        self.__DXL_ADDR_INDIRECTDATA_READ = 224
        self.__DXL_ADDR_PROFILE_VEL = 112
        self.__DXL_ADDR_PROFILE_ACC = 108
        # PID Gain addresses (for position control)
        self.__DXL_ADDR_POSITION_D_GAIN = 80
        self.__DXL_ADDR_POSITION_I_GAIN = 82
        self.__DXL_ADDR_POSITION_P_GAIN = 84
        # Current Limit address
        self.__DXL_ADDR_CURRENT_LIMIT = 38
        self.__DXL_LEN_INDIRECTDATA_READ = 11
        self.__DXL_EXT_POSITION_CTRL = 4
        self.__DXL_VEL_CTRL = 1
        self.__DXL_POSITION_CTRL = 3
        self.__DXL_CURRENT_POSITION_CTRL = 5
        self.__DXL_CURRENT_CTRL = 0
        self.__DXL_POS_UNIT = 0.087891/360.0*2*pi  # unit is radian
        self.__DXL_DEG_POS_UNIT = 0.087891  # unit is degrees
        self.__DXL_VEL_UNIT = 0.229*2*pi/60.0  # unit is radian/s
        self.__DXL_ACC_UNIT = 214.577*2*pi/(60*60)
        self.__DXL_VEL_LIMIT = 75*2*pi/60
        self.__DXL_CURRENT_UNIT = 2.69  # unit is mA
        self.__DXL_CURRENT_LIMIT = 3209.17
        self.__DXL_TORQUE_ENABLE = 1
        self.__DXL_TORQUE_DISABLE = 0
        self.__DXL_ADDR_PRESENT_PWM = 124
        self.__DXL_ADDR_PRESENT_INPUT_VOLTAGE = 144 
        if home_offset is not None:
            self.__DXL_HOME_OFFSET = home_offset

        if isinstance(DXL_PORT, str):
            self.__dxl_portHandler = dynamixel_sdk.PortHandler(self.__DXL_PORT)
            try:
                self.__dxl_portHandler.openPort()
            except:
                rospy.logwarn("Failed to open the port")
                quit()
        else:
            self.__dxl_portHandler = DXL_PORT
        self.__dxl_packetHandler = dynamixel_sdk.PacketHandler(
            self.__DXL_PROTOCOL_VERSION)

        self.__dxl_groupBulkRead = dynamixel_sdk.GroupBulkRead(
            self.__dxl_portHandler, self.__dxl_packetHandler)
        self.__dxl_groupBulkWrite = dynamixel_sdk.GroupBulkWrite(
            self.__dxl_portHandler, self.__dxl_packetHandler)

        # DXL motor setup
        try:
            self.__dxl_portHandler.setBaudRate(self.__DXL_BAUDRATE)
        except:
            rospy.logwarn("Failed to change the baudrate")
            quit()

        rospy.sleep(1.0)
        for motor_index in range(len(self.__DXL_IDs)):
            self.__dxl_packetHandler.write1ByteTxRx(
                self.__dxl_portHandler, self.__DXL_IDs[motor_index], self.__DXL_ADDR_TORQUE_ENABLE, self.__DXL_TORQUE_DISABLE)
            self.__dxl_packetHandler.write1ByteTxRx(
                self.__dxl_portHandler, self.__DXL_IDs[motor_index], self.__DXL_ADDR_DRIVE_MODE, self.__DXL_DIRs[motor_index])
            if home_offset is not None:
                # use position control mode to make the current position inside the single-turn range
                self.__dxl_packetHandler.write1ByteTxRx(
                    self.__dxl_portHandler, self.__DXL_IDs[motor_index], self.__DXL_ADDR_OPERATING_MODE, self.__DXL_POSITION_CTRL)
                self.__dxl_packetHandler.write4ByteTxRx(
                    self.__dxl_portHandler, self.__DXL_IDs[motor_index], self.__DXL_ADDR_HOME_OFFSET, self.__DXL_HOME_OFFSET[motor_index])
            if group_read:
                # indirect data storage for position, current, velocity and temp address
                self.__dxl_packetHandler.write2ByteTxRx(
                    self.__dxl_portHandler, self.__DXL_IDs[motor_index],
                    self.__DXL_ADDR_INDIRECTADDR_READ+0, self.__DXL_ADDR_PRESENT_POS+0)
                self.__dxl_packetHandler.write2ByteTxRx(
                    self.__dxl_portHandler, self.__DXL_IDs[motor_index],
                    self.__DXL_ADDR_INDIRECTADDR_READ+2, self.__DXL_ADDR_PRESENT_POS+1)
                self.__dxl_packetHandler.write2ByteTxRx(
                    self.__dxl_portHandler, self.__DXL_IDs[motor_index],
                    self.__DXL_ADDR_INDIRECTADDR_READ+4, self.__DXL_ADDR_PRESENT_POS+2)
                self.__dxl_packetHandler.write2ByteTxRx(
                    self.__dxl_portHandler, self.__DXL_IDs[motor_index],
                    self.__DXL_ADDR_INDIRECTADDR_READ+6, self.__DXL_ADDR_PRESENT_POS+3)
                self.__dxl_packetHandler.write2ByteTxRx(
                    self.__dxl_portHandler, self.__DXL_IDs[motor_index],
                    self.__DXL_ADDR_INDIRECTADDR_READ+8, self.__DXL_ADDR_PRESENT_CURRENT+0)
                self.__dxl_packetHandler.write2ByteTxRx(
                    self.__dxl_portHandler, self.__DXL_IDs[motor_index],
                    self.__DXL_ADDR_INDIRECTADDR_READ+10, self.__DXL_ADDR_PRESENT_CURRENT+1)
                self.__dxl_packetHandler.write2ByteTxRx(
                    self.__dxl_portHandler, self.__DXL_IDs[motor_index],
                    self.__DXL_ADDR_INDIRECTADDR_READ+12, self.__DXL_ADDR_PRESENT_TEMP)
                self.__dxl_packetHandler.write2ByteTxRx(
                    self.__dxl_portHandler, self.__DXL_IDs[motor_index],
                    self.__DXL_ADDR_INDIRECTADDR_READ+14, self.__DXL_ADDR_PRESENT_VEL+0)
                self.__dxl_packetHandler.write2ByteTxRx(
                    self.__dxl_portHandler, self.__DXL_IDs[motor_index],
                    self.__DXL_ADDR_INDIRECTADDR_READ+16, self.__DXL_ADDR_PRESENT_VEL+1)
                self.__dxl_packetHandler.write2ByteTxRx(
                    self.__dxl_portHandler, self.__DXL_IDs[motor_index],
                    self.__DXL_ADDR_INDIRECTADDR_READ+18, self.__DXL_ADDR_PRESENT_VEL+2)
                self.__dxl_packetHandler.write2ByteTxRx(
                    self.__dxl_portHandler, self.__DXL_IDs[motor_index],
                    self.__DXL_ADDR_INDIRECTADDR_READ+20, self.__DXL_ADDR_PRESENT_VEL+3)   
                self.__dxl_packetHandler.write2ByteTxRx(
                    self.__dxl_portHandler, self.__DXL_IDs[motor_index],
                    self.__DXL_ADDR_INDIRECTADDR_READ+22,self.__DXL_ADDR_PRESENT_PWM+0)
                self.__dxl_packetHandler.write2ByteTxRx(
                    self.__dxl_portHandler, self.__DXL_IDs[motor_index],
                    self.__DXL_ADDR_INDIRECTADDR_READ+24,self.__DXL_ADDR_PRESENT_PWM+1)
                self.__dxl_packetHandler.write2ByteTxRx(
                    self.__dxl_portHandler, self.__DXL_IDs[motor_index],
                    self.__DXL_ADDR_INDIRECTADDR_READ+26,self.__DXL_ADDR_PRESENT_INPUT_VOLTAGE+0)
                self.__dxl_packetHandler.write2ByteTxRx(
                    self.__dxl_portHandler, self.__DXL_IDs[motor_index],
                    self.__DXL_ADDR_INDIRECTADDR_READ+28,self.__DXL_ADDR_PRESENT_INPUT_VOLTAGE+1)
                

            self.__dxl_packetHandler.write1ByteTxRx(
                self.__dxl_portHandler, self.__DXL_IDs[motor_index], self.__DXL_ADDR_OPERATING_MODE, self.__DXL_VEL_CTRL)
            self.__dxl_packetHandler.write1ByteTxRx(
                self.__dxl_portHandler, self.__DXL_IDs[motor_index], self.__DXL_ADDR_TORQUE_ENABLE, self.__DXL_TORQUE_ENABLE)

        for motor_id in self.__DXL_IDs:
            dxl_addparam_result = self.__dxl_groupBulkRead.addParam(
                motor_id, self.__DXL_ADDR_PRESENT_POS, 4)
            if dxl_addparam_result != True:
                print("[Motor ID: %03d] groupBulkRead addparam failed" % motor_id)
                quit()
            # dxl_addparam_result = self.__dxl_groupBulkWrite.addParam(
            #     motor_id, self.__DXL_ADDR_GOAL_VEL, 4,
            #     [dynamixel_sdk.DXL_LOBYTE(dynamixel_sdk.DXL_LOWORD(0)), dynamixel_sdk.DXL_HIBYTE(dynamixel_sdk.DXL_LOWORD(0)),
            #      dynamixel_sdk.DXL_LOBYTE(dynamixel_sdk.DXL_HIWORD(0)), dynamixel_sdk.DXL_HIBYTE(dynamixel_sdk.DXL_HIWORD(0))])
            # if dxl_addparam_result != True:
            #     print("[Motor ID: %03d] groupBulkWrite addparam failed" % motor_id)
            #     quit()
        # self.__dxl_groupBulkWrite.clearParam()

        self.operating_mode = 1  # default mode is velocity control mode
        self.support_operating_mode = [
            self.__DXL_EXT_POSITION_CTRL, self.__DXL_VEL_CTRL, self.__DXL_CURRENT_POSITION_CTRL, self.__DXL_CURRENT_CTRL]
        self.finger_name = finger_name
        self.motor_limits = motor_limits
        self.motor_deg_limits = None
        self.forward_model = forward_model
        self.backward_model = backward_model
        self.forward_model_const = forward_model_const
        self.backward_model_const = backward_model_const

        self.finger_pose = FingerMeasure()
        self.motor_status = MotorMonitor()

        self.finger_pos_sub = rospy.Subscriber(
            "finger_" + self.finger_name+"/measure", FingerMeasure,
            self.__finger_pose_callback)
        self.motor_pos_traj_pub = rospy.Publisher(
            "finger_" + self.finger_name+"/motor_pos_traj", MotorPosTraj,
            queue_size=1)
        self.motor_vel_cmd_pub = rospy.Publisher(
            "finger_" + self.finger_name + "/motor_vel_cmd", MotorPosCmd, queue_size=1)

    def __finger_pose_callback(self, msg):
        """
        Keyword Arguments:
        msg -- finger pos message
        :type msg: FingerMeasure
        """
        self.finger_pose.angles = msg.angles

    def deactive_motors(self, id = None):
        """
        """
        if id:
            self.__dxl_packetHandler.write1ByteTxRx(
                self.__dxl_portHandler, self.__DXL_IDs[id%11], self.__DXL_ADDR_TORQUE_ENABLE, self.__DXL_TORQUE_DISABLE)   
            return         
        for motor_id in self.__DXL_IDs:
            self.__dxl_packetHandler.write1ByteTxRx(
                self.__dxl_portHandler, motor_id, self.__DXL_ADDR_TORQUE_ENABLE, self.__DXL_TORQUE_DISABLE)


    def init_motors(self):
        """
        Keyword Arguments:
        motors_pos -- [motor1_pos, motor2_pos, motor3_pos]
        """
        for motor_index in range(len(self.__DXL_IDs)):
            self.__dxl_packetHandler.write1ByteTxRx(
                self.__dxl_portHandler, self.__DXL_IDs[motor_index], self.__DXL_ADDR_TORQUE_ENABLE, self.__DXL_TORQUE_DISABLE)
            # use position control mode to make the current position inside the single-turn range
            self.__dxl_packetHandler.write1ByteTxRx(
                self.__dxl_portHandler, self.__DXL_IDs[motor_index], self.__DXL_ADDR_OPERATING_MODE, self.__DXL_POSITION_CTRL)
            self.__dxl_packetHandler.write4ByteTxRx(
                self.__dxl_portHandler, self.__DXL_IDs[motor_index], self.__DXL_ADDR_HOME_OFFSET, 0)
            motor_pos, _, _ = self.__dxl_packetHandler.read4ByteTxRx(
                self.__dxl_portHandler, self.__DXL_IDs[motor_index], self.__DXL_ADDR_PRESENT_POS)
            if self.__DXL_DIRs[motor_index] == 0:
                motor_pos = -motor_pos
            print(motor_pos)
            self.__dxl_packetHandler.write4ByteTxRx(
                self.__dxl_portHandler, self.__DXL_IDs[motor_index], self.__DXL_ADDR_HOME_OFFSET, motor_pos)
            self.__dxl_packetHandler.write1ByteTxRx(
                self.__dxl_portHandler, self.__DXL_IDs[motor_index], self.__DXL_ADDR_OPERATING_MODE, self.__DXL_VEL_CTRL)
            self.__dxl_packetHandler.write1ByteTxRx(
                self.__dxl_portHandler, self.__DXL_IDs[motor_index], self.__DXL_ADDR_TORQUE_ENABLE, self.__DXL_TORQUE_ENABLE)

    def init_cable_motors(self):
        """Only works for index and thumb that have hinge motor
        """
        for motor_index in range(len(self.__DXL_IDs)-1):
            self.__dxl_packetHandler.write1ByteTxRx(
                self.__dxl_portHandler, self.__DXL_IDs[motor_index], self.__DXL_ADDR_TORQUE_ENABLE, self.__DXL_TORQUE_DISABLE)
            # use position control mode to make the current position inside the single-turn range
            self.__dxl_packetHandler.write1ByteTxRx(
                self.__dxl_portHandler, self.__DXL_IDs[motor_index], self.__DXL_ADDR_OPERATING_MODE, self.__DXL_POSITION_CTRL)
            self.__dxl_packetHandler.write4ByteTxRx(
                self.__dxl_portHandler, self.__DXL_IDs[motor_index], self.__DXL_ADDR_HOME_OFFSET, 0)
            motor_pos, _, _ = self.__dxl_packetHandler.read4ByteTxRx(
                self.__dxl_portHandler, self.__DXL_IDs[motor_index], self.__DXL_ADDR_PRESENT_POS)
            if self.__DXL_DIRs[motor_index] == 0:
                motor_pos = -motor_pos
            print(motor_pos)
            self.__dxl_packetHandler.write4ByteTxRx(
                self.__dxl_portHandler, self.__DXL_IDs[motor_index], self.__DXL_ADDR_HOME_OFFSET, motor_pos)
            self.__dxl_packetHandler.write1ByteTxRx(
                self.__dxl_portHandler, self.__DXL_IDs[motor_index], self.__DXL_ADDR_OPERATING_MODE, self.__DXL_VEL_CTRL)
            self.__dxl_packetHandler.write1ByteTxRx(
                self.__dxl_portHandler, self.__DXL_IDs[motor_index], self.__DXL_ADDR_TORQUE_ENABLE, self.__DXL_TORQUE_ENABLE)

    def switch_motor_operating_mode(self, mode):
        """
        Keyword Arguments:
        mode -- 1 (velocity control mode), 4 (extended position control mode)
        """
        assert self.support_operating_mode.count(mode)
        for motor_id in self.__DXL_IDs:
            self.__dxl_packetHandler.write1ByteTxRx(
                self.__dxl_portHandler, motor_id, self.__DXL_ADDR_TORQUE_ENABLE, self.__DXL_TORQUE_DISABLE)
            dxl_comm_result, dxl_error = self.__dxl_packetHandler.write1ByteTxRx(
                self.__dxl_portHandler, motor_id, self.__DXL_ADDR_OPERATING_MODE, mode)
            if dxl_comm_result != dynamixel_sdk.COMM_SUCCESS:
                # print(dxl_comm_result)
                print('Failed to switch operating mode', dxl_comm_result, dxl_error)
            self.__dxl_packetHandler.write1ByteTxRx(
                self.__dxl_portHandler, motor_id, self.__DXL_ADDR_TORQUE_ENABLE, self.__DXL_TORQUE_ENABLE)

    def switch_single_motor_operating_mode(self, motor_id, mode):
        """
        Keyword Arguments:
        motor_id --
        mode     --
        """
        id = self.__DXL_IDs[motor_id%11]
        assert self.support_operating_mode.count(mode)
        self.__dxl_packetHandler.write1ByteTxRx(
            self.__dxl_portHandler, id, self.__DXL_ADDR_TORQUE_ENABLE, self.__DXL_TORQUE_DISABLE)
        dxl_comm_result, dxl_error = self.__dxl_packetHandler.write1ByteTxRx(
            self.__dxl_portHandler, id, self.__DXL_ADDR_OPERATING_MODE, mode)
        if dxl_comm_result != dynamixel_sdk.COMM_SUCCESS:
            # print(dxl_comm_result)
            print('Failed to switch operating mode', dxl_comm_result, dxl_error)
        self.__dxl_packetHandler.write1ByteTxRx(
            self.__dxl_portHandler, id, self.__DXL_ADDR_TORQUE_ENABLE, self.__DXL_TORQUE_ENABLE)

    def set_velocity_profile(self, motors_vel):
        """
        This can only applied to position control
        Keyword Arguments:
        motors_vel -- [vel1, vel2, vel3]
        """
        for i in range(len(motors_vel)):
            vel_profile = int(motors_vel[i]/self.__DXL_VEL_UNIT)
            dxl_comm_result, dxl_error = self.__dxl_packetHandler.write4ByteTxRx(
                self.__dxl_portHandler, self.__DXL_IDs[i], self.__DXL_ADDR_PROFILE_VEL, vel_profile)
            if dxl_comm_result != dynamixel_sdk.COMM_SUCCESS:
                print('Failed to set velocity profile')

    def set_acceleration_profile(self, motors_a):
        """
        This can only applied to position control
        Keyword Arguments:
        motors_vel -- [vel1, vel2, vel3]
        """
        for i in range(len(motors_a)):
            vel_profile = int(motors_a[i]/self.__DXL_VEL_UNIT)
            dxl_comm_result, dxl_error = self.__dxl_packetHandler.write4ByteTxRx(
                self.__dxl_portHandler, self.__DXL_IDs[i], self.__DXL_ADDR_PROFILE_VEL, vel_profile)
            if dxl_comm_result != dynamixel_sdk.COMM_SUCCESS:
                print('Failed to set velocity profile')

    def set_pid_gains(self, kp=800, ki=100, kd=1000):
        """
        Set PID gains for position control on all motors.

        Keyword Arguments:
        kp -- Position P Gain (default: 800)
        ki -- Position I Gain (default: 100)
        kd -- Position D Gain (default: 1000)

        Note: Torque must be disabled to write PID gains.
        """
        print(f"Setting PID gains: Kp={kp}, Ki={ki}, Kd={kd}")
        for motor_id in self.__DXL_IDs:
            # Disable torque to allow writing PID gains
            self.__dxl_packetHandler.write1ByteTxRx(
                self.__dxl_portHandler, motor_id, self.__DXL_ADDR_TORQUE_ENABLE, self.__DXL_TORQUE_DISABLE)

            # Write P Gain
            dxl_comm_result, dxl_error = self.__dxl_packetHandler.write2ByteTxRx(
                self.__dxl_portHandler, motor_id, self.__DXL_ADDR_POSITION_P_GAIN, kp)
            if dxl_comm_result != dynamixel_sdk.COMM_SUCCESS:
                print(f'Failed to set P Gain for motor {motor_id}')

            # Write I Gain
            dxl_comm_result, dxl_error = self.__dxl_packetHandler.write2ByteTxRx(
                self.__dxl_portHandler, motor_id, self.__DXL_ADDR_POSITION_I_GAIN, ki)
            if dxl_comm_result != dynamixel_sdk.COMM_SUCCESS:
                print(f'Failed to set I Gain for motor {motor_id}')

            # Write D Gain
            dxl_comm_result, dxl_error = self.__dxl_packetHandler.write2ByteTxRx(
                self.__dxl_portHandler, motor_id, self.__DXL_ADDR_POSITION_D_GAIN, kd)
            if dxl_comm_result != dynamixel_sdk.COMM_SUCCESS:
                print(f'Failed to set D Gain for motor {motor_id}')

            # Re-enable torque
            self.__dxl_packetHandler.write1ByteTxRx(
                self.__dxl_portHandler, motor_id, self.__DXL_ADDR_TORQUE_ENABLE, self.__DXL_TORQUE_ENABLE)

        print(f"PID gains set successfully for all {len(self.__DXL_IDs)} motors")

    def set_single_motor_pid_gains(self, motor_id, kp, ki, kd):
        """
        Set PID gains for a single motor.

        Args:
            motor_id: Motor ID (e.g., 11, 12, 13, 14)
            kp: Position P Gain
            ki: Position I Gain
            kd: Position D Gain
        """
        print(f"Setting PID for motor {motor_id}: Kp={kp}, Ki={ki}, Kd={kd}")

        # Disable torque
        self.__dxl_packetHandler.write1ByteTxRx(
            self.__dxl_portHandler, motor_id, self.__DXL_ADDR_TORQUE_ENABLE, self.__DXL_TORQUE_DISABLE)

        # Write PID gains
        self.__dxl_packetHandler.write2ByteTxRx(
            self.__dxl_portHandler, motor_id, self.__DXL_ADDR_POSITION_P_GAIN, kp)
        self.__dxl_packetHandler.write2ByteTxRx(
            self.__dxl_portHandler, motor_id, self.__DXL_ADDR_POSITION_I_GAIN, ki)
        self.__dxl_packetHandler.write2ByteTxRx(
            self.__dxl_portHandler, motor_id, self.__DXL_ADDR_POSITION_D_GAIN, kd)

        # Re-enable torque
        self.__dxl_packetHandler.write1ByteTxRx(
            self.__dxl_portHandler, motor_id, self.__DXL_ADDR_TORQUE_ENABLE, self.__DXL_TORQUE_ENABLE)

    def get_pid_gains(self):
        """
        Read current PID gains from all motors for verification.

        Returns:
            dict: {motor_id: {'kp': value, 'ki': value, 'kd': value}, ...}
        """
        pid_gains = {}
        for motor_id in self.__DXL_IDs:
            kp, _, _ = self.__dxl_packetHandler.read2ByteTxRx(
                self.__dxl_portHandler, motor_id, self.__DXL_ADDR_POSITION_P_GAIN)
            ki, _, _ = self.__dxl_packetHandler.read2ByteTxRx(
                self.__dxl_portHandler, motor_id, self.__DXL_ADDR_POSITION_I_GAIN)
            kd, _, _ = self.__dxl_packetHandler.read2ByteTxRx(
                self.__dxl_portHandler, motor_id, self.__DXL_ADDR_POSITION_D_GAIN)
            pid_gains[motor_id] = {'kp': kp, 'ki': ki, 'kd': kd}
            print(f"Motor {motor_id}: Kp={kp}, Ki={ki}, Kd={kd}")
        return pid_gains

    def set_goal_current(self, goal_current_mA):
        """
        Set Goal Current for all motors (used in Current-Based Position Control Mode).

        Args:
            goal_current_mA: Goal current in mA (will be applied as limit in Mode 5)
        """
        # Convert mA to raw value
        goal_current_raw = int(goal_current_mA / self.__DXL_CURRENT_UNIT)
        print(f"Setting Goal Current: {goal_current_mA} mA (raw: {goal_current_raw})")

        for motor_id in self.__DXL_IDs:
            dxl_comm_result, dxl_error = self.__dxl_packetHandler.write2ByteTxRx(
                self.__dxl_portHandler, motor_id, self.__DXL_ADDR_GOAL_CURRENT, goal_current_raw)
            if dxl_comm_result != dynamixel_sdk.COMM_SUCCESS:
                print(f'Failed to set Goal Current for motor {motor_id}')

        print(f"Goal Current set for all {len(self.__DXL_IDs)} motors")

    def get_goal_current(self):
        """
        Read Goal Current from all motors.
        """
        for motor_id in self.__DXL_IDs:
            goal_current_raw, _, _ = self.__dxl_packetHandler.read2ByteTxRx(
                self.__dxl_portHandler, motor_id, self.__DXL_ADDR_GOAL_CURRENT)
            # Handle signed 16-bit value
            if goal_current_raw > 32767:
                goal_current_raw -= 65536
            goal_current_mA = goal_current_raw * self.__DXL_CURRENT_UNIT
            print(f"Motor {motor_id}: Goal Current = {goal_current_mA:.1f} mA (raw: {goal_current_raw})")

    def get_motor_operating_mode(self):
        """
        Keyword Arguments:
        """
        motors_mode = []
        for motor_id in self.__DXL_IDs:
            motor_mode, _, _ = self.__dxl_packetHandler.read1ByteTxRx(
                self.__dxl_portHandler, motor_id, self.__DXL_ADDR_OPERATING_MODE)
            motors_mode.append(motor_mode)
        return motors_mode

    def send_motor_pos_cmd(self, motors_pos_traj, reverse = None, deg = None):
        """
        Keyword Arguments:
        motors_pos_traj -- [motor1_pos, motor2_pos, motor3_pos]
        Beware of reverse, assumes 5 motors
        """

        if deg:
            angle_unit = self.__DXL_DEG_POS_UNIT
        else:
            angle_unit = self.__DXL_POS_UNIT

        if reverse:
            ids = [4,3,2,1,0]
        else:
            ids = [0,1,2,3,4]


        for i in ids:
            if self.motor_limits is not None:
                if motors_pos_traj[i] > self.motor_limits[i][1]:
                    motors_pos_traj[i] = (self.motor_limits[i][1])   #/self.__DXL_POS_UNIT

                if motors_pos_traj[i] < self.motor_limits[i][0]:
                    motors_pos_traj[i] = (self.motor_limits[i][0])   #/self.__DXL_POS_UNIT
            motor_pos = int(motors_pos_traj[i]/angle_unit)

            
            # self.__dxl_packetHandler.write4ByteTxRx(
            #     self.__dxl_portHandler, self.__DXL_IDs[i], self.__DXL_ADDR_GOAL_POS, motor_pos)
            goal_pos = [
                dynamixel_sdk.DXL_LOBYTE(dynamixel_sdk.DXL_LOWORD(motor_pos)),
                dynamixel_sdk.DXL_HIBYTE(dynamixel_sdk.DXL_LOWORD(motor_pos)),
                dynamixel_sdk.DXL_LOBYTE(dynamixel_sdk.DXL_HIWORD(motor_pos)),
                dynamixel_sdk.DXL_HIBYTE(dynamixel_sdk.DXL_HIWORD(motor_pos))]
            self.__dxl_groupBulkWrite.addParam(
                self.__DXL_IDs[i], self.__DXL_ADDR_GOAL_POS, 4, goal_pos)
        dxl_comm_result = self.__dxl_groupBulkWrite.txPacket()
        if dxl_comm_result != dynamixel_sdk.COMM_SUCCESS:
            return False
        self.__dxl_groupBulkWrite.clearParam()
        return True
    
    def send_single_motor_pos_cmd(self, motors_pos_traj, id):
        angle_unit = self.__DXL_POS_UNIT

        if self.motor_limits is not None:
            if motors_pos_traj > self.motor_limits[id%11][1]:
                motors_pos_traj = (self.motor_limits[id%11][1])

            if motors_pos_traj < self.motor_limits[id%11][0]:
                motors_pos_traj = self.motor_limits[id%11][0]
        motor_pos = int(motors_pos_traj/angle_unit)

        
        # self.__dxl_packetHandler.write4ByteTxRx(
        #     self.__dxl_portHandler, self.__DXL_IDs[i], self.__DXL_ADDR_GOAL_POS, motor_pos)
        goal_pos = [
            dynamixel_sdk.DXL_LOBYTE(dynamixel_sdk.DXL_LOWORD(motor_pos)),
            dynamixel_sdk.DXL_HIBYTE(dynamixel_sdk.DXL_LOWORD(motor_pos)),
            dynamixel_sdk.DXL_LOBYTE(dynamixel_sdk.DXL_HIWORD(motor_pos)),
            dynamixel_sdk.DXL_HIBYTE(dynamixel_sdk.DXL_HIWORD(motor_pos))]
        self.__dxl_groupBulkWrite.addParam(
            self.__DXL_IDs[id%11], self.__DXL_ADDR_GOAL_POS, 4, goal_pos)
        dxl_comm_result = self.__dxl_groupBulkWrite.txPacket()
        if dxl_comm_result != dynamixel_sdk.COMM_SUCCESS:
            return False
        self.__dxl_groupBulkWrite.clearParam()
        return True


    def send_motor_vel_cmd(self, motors_vel):
        """
        Keyword Arguments:
        motors_vel -- [motor1_vel, motor2_vel, motor3_vel]
        """
        for i in range(len(self.__DXL_IDs)):
            if motors_vel[i] > self.__DXL_VEL_LIMIT:
                motors_vel[i] = self.__DXL_VEL_LIMIT
            if motors_vel[i] < -self.__DXL_VEL_LIMIT:
                motors_vel[i] = -self.__DXL_VEL_LIMIT
            motor_vel = int(motors_vel[i] / self.__DXL_VEL_UNIT)
            # self.__dxl_packetHandler.write4ByteTxRx(
            #     self.__dxl_portHandler, self.__DXL_IDs[i], self.__DXL_ADDR_GOAL_VEL, motor_vel)
            goal_vel = [
                dynamixel_sdk.DXL_LOBYTE(dynamixel_sdk.DXL_LOWORD(motor_vel)),
                dynamixel_sdk.DXL_HIBYTE(dynamixel_sdk.DXL_LOWORD(motor_vel)),
                dynamixel_sdk.DXL_LOBYTE(dynamixel_sdk.DXL_HIWORD(motor_vel)),
                dynamixel_sdk.DXL_HIBYTE(dynamixel_sdk.DXL_HIWORD(motor_vel))]
            self.__dxl_groupBulkWrite.addParam(
                self.__DXL_IDs[i], self.__DXL_ADDR_GOAL_VEL, 4, goal_vel)
        dxl_comm_result = self.__dxl_groupBulkWrite.txPacket()
        if dxl_comm_result != dynamixel_sdk.COMM_SUCCESS:
            return False
        self.__dxl_groupBulkWrite.clearParam()
        return True

    def send_motor_torque_cmd(self, motor_id, torque):
        """
        Keyword Arguments:
        motor_id --
        torque   --
        """
        id = self.__DXL_IDs[motor_id%11]
        motor_torque = int(torque/self.__DXL_CURRENT_UNIT)
        dxl_comm_result, dxl_error = self.__dxl_packetHandler.write2ByteTxRx(
            self.__dxl_portHandler, id, self.__DXL_ADDR_GOAL_CURRENT, motor_torque)
        if dxl_comm_result != dynamixel_sdk.COMM_SUCCESS:
            print('Failed to send torque command', dxl_error, "  ", dxl_comm_result, " ", self.__dxl_portHandler.is_open)
            return False
        return True
    
    def send_torque_sync_cmd(self, torque_profile):
        """
        Send torques to all motors using GroupSyncWrite.
        
        Keyword Arguments:
        torque_profile -- [t1, t2, t3, ...]
                        List of torques corresponding to self.__DXL_IDs order.
        """
        # Create GroupSyncWrite instance for Goal Current (2 bytes)
        groupSyncWrite = dynamixel_sdk.GroupSyncWrite(
            self.__dxl_portHandler,
            self.__dxl_packetHandler,
            self.__DXL_ADDR_GOAL_CURRENT,
            2  # length of Goal Current register
        )

        for i, torque in enumerate(torque_profile):
            # Clamp to current limits
            if torque > self.__DXL_CURRENT_LIMIT:
                torque = self.__DXL_CURRENT_LIMIT
            if torque < -self.__DXL_CURRENT_LIMIT:
                torque = -self.__DXL_CURRENT_LIMIT

            # Convert to Dynamixel units
            motor_torque = int(torque / self.__DXL_CURRENT_UNIT)

            # Convert to 2-byte array [low, high]
            param_goal_current = [
                dynamixel_sdk.DXL_LOBYTE(motor_torque),
                dynamixel_sdk.DXL_HIBYTE(motor_torque)
            ]

            # Add parameter for this motor
            motor_id = self.__DXL_IDs[i]
            groupSyncWrite.addParam(motor_id, param_goal_current)

        # Transmit the packet once to update all motors together
        dxl_comm_result = groupSyncWrite.txPacket()
        if dxl_comm_result != dynamixel_sdk.COMM_SUCCESS:
            print("GroupSyncWrite failed:", dxl_comm_result)
            return False

        # Clear buffer for next use
        groupSyncWrite.clearParam()
        return True
    
    def get_motor_pos(self):
        """
        """
        motors_pos = []
        # for motor_id in self.__DXL_IDs:
        #     motor_pos, _, _ = self.__dxl_packetHandler.read4ByteTxRx(
        #         self.__dxl_portHandler, motor_id, self.__DXL_ADDR_PRESENT_POS)
        #     if motor_pos > 0x7fffffff:
        #         motor_pos = motor_pos - 4294967296
        #     motors_pos.append(motor_pos * self.__DXL_POS_UNIT)

        # accelerate reading using group bulk reading
        dxl_comm_result = self.__dxl_groupBulkRead.txRxPacket()
        if dxl_comm_result != dynamixel_sdk.COMM_SUCCESS:
            rospy.logwarn(str(dxl_comm_result))
            return False
            # return str(dxl_comm_result)          #  dxl_comm_result
        for motor_id in self.__DXL_IDs:
            motor_pos = self.__dxl_groupBulkRead.getData(
                motor_id, self.__DXL_ADDR_PRESENT_POS, 4)
            if motor_pos > 0x7fffffff:
                motor_pos = motor_pos - 4294967296
            motors_pos.append(motor_pos * self.__DXL_POS_UNIT)
        self.motor_status.motorsPos = motors_pos
        return motors_pos
    def get_baud(self):
        BAUD_ADDR = 8
        BAUD_LEN = 1

        for motor_id in self.__DXL_IDs:
            baud_val, dxl_comm_result, dxl_error = self.__dxl_packetHandler.read1ByteTxRx(
                self.__dxl_portHandler, motor_id, BAUD_ADDR
            )
            if dxl_comm_result != dynamixel_sdk.COMM_SUCCESS:
                print(f"Failed to read baudrate from motor {motor_id}")
            else:
                print(f"Motor {motor_id} baud setting: {baud_val}")
    
    def get_motor_vel(self):
        
        motors_vel = []
        dxl_comm_result = self.__dxl_groupBulkRead.txRxPacket()
        if dxl_comm_result != dynamixel_sdk.COMM_SUCCESS:
            rospy.logwarn(str(dxl_comm_result))
            return False
            # return str(dxl_comm_result)          #  dxl_comm_result
        for motor_id in self.__DXL_IDs:
            motor_vel = self.__dxl_groupBulkRead.getData(
                motor_id, self.__DXL_ADDR_PRESENT_VEL, 4)
            if motor_vel > 0x7fffffff:
                motor_vel = motor_vel - 4294967296
            motors_vel.append(motor_vel * self.__DXL_VEL_UNIT)

        self.motor_status.motorsVel = motors_vel
        # print(motors_vel, "here")
        return motors_vel

    def get_motor_torque(self):
        """
        Read Present Current (torque) from all motors using GroupSyncRead.
        """
        ADDR_PRESENT_CURRENT = self.__DXL_ADDR_PRESENT_CURRENT
        LEN_PRESENT_CURRENT = 2  # 2 bytes

        # Create GroupSyncRead instance
        groupSyncRead = dynamixel_sdk.GroupSyncRead(
            self.__dxl_portHandler,
            self.__dxl_packetHandler,
            ADDR_PRESENT_CURRENT,
            LEN_PRESENT_CURRENT
        )

        # Add all motor IDs once
        for motor_id in self.__DXL_IDs:
            dxl_addparam_result = groupSyncRead.addParam(motor_id)
            if not dxl_addparam_result:
                print(f"Failed to add motor ID {motor_id} to GroupSyncRead")
                return None

        # Send the sync read packet
        dxl_comm_result = groupSyncRead.txRxPacket()
        if dxl_comm_result != dynamixel_sdk.COMM_SUCCESS:
            print("GroupSyncRead failed:", dxl_comm_result)
            return None

        # Collect results
        motors_torque = []
        for motor_id in self.__DXL_IDs:
            if groupSyncRead.isAvailable(motor_id, ADDR_PRESENT_CURRENT, LEN_PRESENT_CURRENT):
                torque_val = groupSyncRead.getData(motor_id, ADDR_PRESENT_CURRENT, LEN_PRESENT_CURRENT)
                # Convert signed 16-bit value
                if torque_val > 32767:
                    torque_val -= 65536
                motors_torque.append(torque_val)
            else:
                print(f"No data available for motor ID {motor_id}")
                motors_torque.append(0)

        # Update status
        self.motor_status.torque = motors_torque

        # Clear params for next read
        groupSyncRead.clearParam()

        return motors_torque
    
    def get_single_motor_torque(self, motor_id):
        motor_torque, _, _ = self.__dxl_packetHandler.read2ByteTxRx(
            self.__dxl_portHandler, motor_id, self.__DXL_ADDR_PRESENT_CURRENT)
        if motor_torque > 32767:
            motor_torque = motor_torque - 65536
        self.motor_status.torque = motor_torque
        return motor_torque
    
    def update_motor_pos(self, motors_pos):
        """
        Keyword Arguments:
        motors_pos -- []
        """
        if len(motors_pos) < 3:
            motors_pos.append(0)
        self.motor_status.motorsPos = motors_pos

    def update_motor_vel(self, motors_vel):
        """
        Keyword Arguments:
        motors_pos -- []
        """
        if len(motors_vel) < 3:
            motors_vel.append(0)
        self.motor_status.motorsVel = motors_vel

    def update_motor_current(self, motors_current):
        """
        Keyword Arguments:
        motors_current -- []
        """
        if len(motors_current) < 3:
            motors_current.append(0)
        self.motor_status.torque = motors_current

    def update_motor_temp(self, motors_temp):
        """
        Keyword Arguments:
        motors_temp -- []
        """
        if len(motors_temp) < 3:
            print("temp")
            motors_temp.append(0)
        self.motor_status.coilsTemp = motors_temp

    def update_motor_pwm(self, motors_pwm):
        """
        Keyword Arguments:
        motors_pwm -- []
        """
        if len(motors_pwm) < 3:
            # print("hereeee")
            motors_pwm.append(0)
        self.motor_status.pwm = motors_pwm

    def update_motor_volts(self, motors_volts):
        """
        Keyword Arguments:
        motors_volts -- []
        """
        if len(motors_volts) < 3:
            # print("hereeee")
            motors_volts.append(0)
        self.motor_status.motorsVolts = motors_volts

    def get_finger_pose(self):
        """Get current finger pose
        """
        return self.finger_pose

    def cal_pose_control_input(self, q_des, gain):
        """
        Keyword Arguments:
        q_des -- 3x1 numpy array
        gain -- 3x3 numpy array
        return control input as 2x1 array
        """
        q_cur = np.array(self.finger_pose.angles)
        q_error = q_des - q_cur
        model_mat = np.zeros((3, 2))
        if q_error[0] >= 0:
            model_mat[0] = self.forward_model[0]
        else:
            model_mat[0] = self.backward_model[0]
        if q_error[1] >= 0:
            model_mat[1] = self.forward_model[1]
        else:
            model_mat[1] = self.backward_model[1]
        if q_error[2] >= 0:
            model_mat[2] = self.forward_model[2]
        else:
            model_mat[2] = self.backward_model[2]

        sol = np.linalg.lstsq(model_mat, np.matmul(gain, q_error), rcond=None)
        u = sol[0]
        # motors_torque = self.get_motor_torque()
        # print("torque")
        # print(motors_torque)
        # print("cmd")
        # print(u)
        motors_pos = self.motor_status.motorsPos
        # print(motors_pos)
        if u[0] < 0:
            # if (motors_torque[0] >= 0) and (motors_torque[0] < 5):
            # if abs(motors_torque[0]) <= 10:
            if motors_pos[0] < -0.6:
                u[0] = 0
        if u[1] < 0:
            # if (motors_torque[1] >= 0) and (motors_torque[1] < 5):
            # if abs(motors_torque[1]) <= 10:
            if motors_pos[1] < -0.5:
                u[1] = 0

        return u
