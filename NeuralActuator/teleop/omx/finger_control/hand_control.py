import dynamixel_sdk
from math import pi

from finger_control.finger_control import FingerControlDXL
import rospy


class HandControlDXL:

    def __init__(self, finger_configuration, DXL_PORT, init_motors=False, init_all_motors=False):
        """Initialize a hand controller
        finger_configuration: dict with keys being 'index', 'middle', 'ring', 'little', 'thumb'
        DXL_PORT: str
        """
        self.__DXL_PORT = DXL_PORT
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
        self.__DXL_ADDR_GOAL_CURRENT = 102
        self.__DXL_ADDR_PRESENT_CURRENT = 126
        self.__DXL_ADDR_PRESENT_TEMP = 146
        self.__DXL_ADDR_INDIRECTADDR_READ = 168
        self.__DXL_ADDR_INDIRECTDATA_READ = 224
        self.__DXL_LEN_INDIRECTDATA_READ = 15
        self.__DXL_EXT_POSITION_CTRL = 4
        self.__DXL_VEL_CTRL = 1
        self.__DXL_CURRENT_POSITION_CTRL = 5
        self.__DXL_POS_UNIT = 0.087891/360.0*2*pi  # unit is radian
        self.__DXL_VEL_UNIT = 0.229*2*pi/60.0  # unit is radian/s
        self.__DXL_VEL_LIMIT = 75*2*pi/60
        self.__DXL_CURRENT_UNIT = 2.69  # unit is mA
        self.__DXL_CURRENT_LIMIT = 3209.17
        self.__DXL_VOLTAGE_UNIT = 0.1
        self.__DXL_TORQUE_ENABLE = 1
        self.__DXL_TORQUE_DISABLE = 0

        self.fingers = {}
        self.finger_motors = {}
        self.finger_motor_limits = {}
        self.finger_list = []
        self.motor_list = []

        self.__dxl_portHandler = dynamixel_sdk.PortHandler(self.__DXL_PORT)
        self.__dxl_packetHandler = dynamixel_sdk.PacketHandler(
            self.__DXL_PROTOCOL_VERSION)

        try:
            self.__dxl_portHandler.openPort()
        except:
            print(self.__dxl_portHandler)
            rospy.logwarn("Failed to open the port")
            quit()

        self.__dxl_groupBulkRead = dynamixel_sdk.GroupBulkRead(
            self.__dxl_portHandler, self.__dxl_packetHandler)
        self.__dxl_groupBulkWrite = dynamixel_sdk.GroupBulkWrite(
            self.__dxl_portHandler, self.__dxl_packetHandler)

        self.__dxl_groupSyncRead = dynamixel_sdk.GroupSyncRead(
            self.__dxl_portHandler, self.__dxl_packetHandler, self.__DXL_ADDR_INDIRECTDATA_READ,
            self.__DXL_LEN_INDIRECTDATA_READ)
        # self.__dxl_groupSyncRead = dynamixel_sdk.GroupSyncRead(
        #     self.__dxl_portHandler, self.__dxl_packetHandler, self.__DXL_ADDR_PRESENT_POS, 4)
        self.__dxl_groupSyncWrite = dynamixel_sdk.GroupSyncWrite(
            self.__dxl_portHandler, self.__dxl_packetHandler, self.__DXL_ADDR_GOAL_VEL, 4)


        for finger_name in finger_configuration:
            self.fingers[finger_name] = FingerControlDXL(
                finger_name, finger_configuration[finger_name]['motor_ids'],
                finger_configuration[finger_name]['motor_dirs'],
                finger_configuration[finger_name]['motor_limits'], self.__dxl_portHandler,
                finger_configuration[finger_name]['forward_model'],
                finger_configuration[finger_name]['backward_model'],
                finger_configuration[finger_name]['forward_model_const'],
                finger_configuration[finger_name]['backward_model_const'],
                finger_configuration[finger_name]['home_offset'], group_read=True)
            # store some commonly used parameters
            self.finger_motors[finger_name] = self.fingers[finger_name].motor_ids
            self.finger_motor_limits[finger_name] = finger_configuration[finger_name]['motor_limits']
            self.finger_list.append(finger_name)
            self.motor_list.extend(self.fingers[finger_name].motor_ids)
            if init_motors:
                if (len(finger_configuration[finger_name]['motor_ids']) > 2) and (init_all_motors == False):
                    self.fingers[finger_name].init_cable_motors()
                else:
                    self.fingers[finger_name].init_motors()

        for motor_id in self.motor_list:
            dxl_addparam_result = self.__dxl_groupBulkRead.addParam(
                motor_id, self.__DXL_ADDR_PRESENT_POS, 4)
            if dxl_addparam_result != True:
                print(
                    "[Motor ID: %03d] groupBulkRead addparam position failed" % motor_id)
                quit()

        for motor_id in self.motor_list:
            dxl_addparam_result = self.__dxl_groupSyncRead.addParam(motor_id)

    def clear_port(self):
        """Clear port
        """
        self.__dxl_portHandler.clearPort()

    def close_port(self):
        """Close port
        """
        self.__dxl_portHandler.closePort()

    def open_port(self):
        """Open port
        """
        print("port is being opened")
        self.__dxl_portHandler.openPort()
        # print("port opened")

    def update_motor_pos_bulk(self):
        """Update positions of all motors
        """
        # now = rospy.get_time()
        dxl_comm_result = self.__dxl_groupBulkRead.txRxPacket()
        # print(f"get time: {rospy.get_time()-now}")
        if dxl_comm_result != dynamixel_sdk.COMM_SUCCESS:
            return False
        for finger_name in self.finger_motors:
            motors_pos = []
            for motor_id in self.finger_motors[finger_name]:
                motor_pos = self.__dxl_groupBulkRead.getData(
                    motor_id, self.__DXL_ADDR_PRESENT_POS, 4)
                if motor_pos > 0x7fffffff:
                    motor_pos = motor_pos - 4294967296
                motors_pos.append(motor_pos * self.__DXL_POS_UNIT)
            self.fingers[finger_name].update_motor_pos(motors_pos)
        return True

    def update_motor_pos_sync(self):
        """Update positions of all motors
        """
        dxl_comm_result = self.__dxl_groupSyncRead.txRxPacket()
        if dxl_comm_result != dynamixel_sdk.COMM_SUCCESS:
            return False
        for finger_name in self.finger_motors:
            motors_pos = []
            for motor_id in self.finger_motors[finger_name]:
                motor_pos = self.__dxl_groupSyncRead.getData(
                    motor_id, self.__DXL_ADDR_PRESENT_POS, 4)
                if motor_pos > 0x7fffffff:
                    motor_pos = motor_pos - 4294967296
                motors_pos.append(motor_pos * self.__DXL_POS_UNIT)
            self.fingers[finger_name].update_motor_pos(motors_pos)
        return True

    def update_motor_status_bulk(self):
        """Update status of all motors
        """
        dxl_comm_result = self.__dxl_groupBulkRead.txRxPacket()
        if dxl_comm_result != dynamixel_sdk.COMM_SUCCESS:
            return False
        for finger_name in self.finger_motors:
            motors_pos = []
            for motor_id in self.finger_motors[finger_name]:
                motor_pos = self.__dxl_groupBulkRead.getData(
                    motor_id, self.__DXL_ADDR_PRESENT_POS, 4)
                if motor_pos > 0x7fffffff:
                    motor_pos = motor_pos - 4294967296
                motors_pos.append(motor_pos * self.__DXL_POS_UNIT)
            self.fingers[finger_name].update_motor_pos(motors_pos)
            motors_current = []
            for motor_id in self.finger_motors[finger_name]:
                motor_current = self.__dxl_groupBulkRead.getData(
                    motor_id, self.__DXL_ADDR_PRESENT_CURRENT, 2)
                motors_current.append(motor_current * self.__DXL_CURRENT_UNIT)
            self.fingers[finger_name].update_motor_current(motors_current)
            motors_temp = []
            for motor_id in self.finger_motors[finger_name]:
                motor_temp = self.__dxl_groupBulkRead.getData(
                    motor_id, self.__DXL_ADDR_PRESENT_TEMP, 1)
                motors_temp.append(motor_temp)
            self.fingers[finger_name].update_motor_temp(motors_temp)

    def update_motor_status_sync(self):
        """Update status of all motors
        """
        dxl_comm_result = self.__dxl_groupSyncRead.txRxPacket()
        if dxl_comm_result != dynamixel_sdk.COMM_SUCCESS:
            return False
        for finger_name in self.finger_motors:
            # POSITION
            motors_pos = []
            for motor_id in self.finger_motors[finger_name]:
                motor_pos = self.__dxl_groupSyncRead.getData(
                    motor_id, self.__DXL_ADDR_INDIRECTDATA_READ, 4)
                if motor_pos > 0x7fffffff:
                    motor_pos = motor_pos - 4294967296
                motors_pos.append(motor_pos * self.__DXL_POS_UNIT)
            self.fingers[finger_name].update_motor_pos(motors_pos)
            # CURRENT
            motors_current = []
            for motor_id in self.finger_motors[finger_name]:
                motor_current = self.__dxl_groupSyncRead.getData(
                    motor_id, self.__DXL_ADDR_INDIRECTDATA_READ+4, 2)
                if motor_current > 32767:
                    motor_current = motor_current - 65536
                motors_current.append(motor_current * self.__DXL_CURRENT_UNIT)
            self.fingers[finger_name].update_motor_current(motors_current)
            # TEMPERATURE
            motors_temp = []
            for motor_id in self.finger_motors[finger_name]:
                motor_temp = self.__dxl_groupSyncRead.getData(
                    motor_id, self.__DXL_ADDR_INDIRECTDATA_READ+6, 1)
                motors_temp.append(motor_temp)
            self.fingers[finger_name].update_motor_temp(motors_temp)
            # VELOCITY
            motors_vel = []
            for motor_id in self.finger_motors[finger_name]:
                motor_vel = self.__dxl_groupSyncRead.getData(
                    motor_id, self.__DXL_ADDR_INDIRECTDATA_READ+7, 4)
                if motor_vel > 0x7fffffff:
                    motor_vel = motor_vel - 4294967296
                motors_vel.append(motor_vel * self.__DXL_VEL_UNIT)
            self.fingers[finger_name].update_motor_vel(motors_vel)
            # PWM
            motors_pwm = []
            for motor_id in self.finger_motors[finger_name]:
                pwm = self.__dxl_groupSyncRead.getData(
                    motor_id, self.__DXL_ADDR_INDIRECTDATA_READ+11, 2)
                if pwm > 32767:
                    pwm -= 65536
                motors_pwm.append(pwm)
                # print(pwm)
            self.fingers[finger_name].update_motor_pwm(motors_pwm)
            # VOLTAGE
            motors_volts = []
            for motor_id in self.finger_motors[finger_name]:
                volts = self.__dxl_groupSyncRead.getData(
                    motor_id, self.__DXL_ADDR_INDIRECTDATA_READ+13, 2)
                if volts > 32767:
                    volts -= 65536
                motors_volts.append(volts*self.__DXL_VOLTAGE_UNIT)
                # print(volts)
            self.fingers[finger_name].update_motor_volts(motors_volts)
            
            
        return True

    def switch_motor_operating_mode(self, mode):
        """
        Keyword Arguments:
        mode -- 1 (velocity control mode), 4 (extended position control mode)
        """
        for finger_name in self.finger_motors:
            self.fingers[finger_name].switch_motor_operating_mode(mode)
        return True

    def send_motor_pos_cmd_bulk(self, finger_motor_pos):
        """
        Keyword Arguments:
        finger_motor_pos -- {'index': [], 'middle': [], ...}
        """
        for finger_name in finger_motor_pos:
            for i in range(len(self.finger_motors[finger_name])):
                if self.finger_motor_limits[finger_name] is not None:
                    if finger_motor_pos[finger_name][i] > self.finger_motor_limits[finger_name][i]:
                        finger_motor_pos[finger_name][i] = self.finger_motor_limits[finger_name][i]
                motor_pos = int(
                    finger_motor_pos[finger_name][i]/self.__DXL_POS_UNIT)
                goal_pos = [
                    dynamixel_sdk.DXL_LOBYTE(
                        dynamixel_sdk.DXL_LOWORD(motor_pos)),
                    dynamixel_sdk.DXL_HIBYTE(
                        dynamixel_sdk.DXL_LOWORD(motor_pos)),
                    dynamixel_sdk.DXL_LOBYTE(
                        dynamixel_sdk.DXL_HIWORD(motor_pos)),
                    dynamixel_sdk.DXL_HIBYTE(
                        dynamixel_sdk.DXL_HIWORD(motor_pos))]
                self.__dxl_groupBulkWrite.addParam(
                    self.finger_motors[finger_name][i], self.__DXL_ADDR_GOAL_POS, 4, goal_pos)

        dxl_comm_result = self.__dxl_groupBulkWrite.txPacket()
        if dxl_comm_result != dynamixel_sdk.COMM_SUCCESS:
            return False
        self.__dxl_groupBulkWrite.clearParam()
        return True

    def send_motor_vel_cmd_bulk(self, finger_motor_vel):
        """
        Keyword Arguments:
        finger_motor_vel -- {'index': [], 'middle': [], ...}
        """
        for finger_name in finger_motor_vel:
            for i in range(len(self.finger_motors[finger_name])):
                if finger_motor_vel[finger_name][i] > self.__DXL_VEL_LIMIT:
                    finger_motor_vel[finger_name][i] = self.__DXL_VEL_LIMIT
                if finger_motor_vel[finger_name][i] < -self.__DXL_VEL_LIMIT:
                    finger_motor_vel[finger_name][i] = -self.__DXL_VEL_LIMIT
                motor_vel = int(
                    finger_motor_vel[finger_name][i] / self.__DXL_VEL_UNIT)
                goal_vel = [
                    dynamixel_sdk.DXL_LOBYTE(
                        dynamixel_sdk.DXL_LOWORD(motor_vel)),
                    dynamixel_sdk.DXL_HIBYTE(
                        dynamixel_sdk.DXL_LOWORD(motor_vel)),
                    dynamixel_sdk.DXL_LOBYTE(
                        dynamixel_sdk.DXL_HIWORD(motor_vel)),
                    dynamixel_sdk.DXL_HIBYTE(
                        dynamixel_sdk.DXL_HIWORD(motor_vel))]
                self.__dxl_groupBulkWrite.addParam(
                    self.finger_motors[finger_name][i], self.__DXL_ADDR_GOAL_VEL, 4, goal_vel)

        dxl_comm_result = self.__dxl_groupBulkWrite.txPacket()
        if dxl_comm_result != dynamixel_sdk.COMM_SUCCESS:
            return False
        self.__dxl_groupBulkWrite.clearParam()
        return True

    def send_motor_vel_cmd_sync(self, finger_motor_vel):
        """
        Keyword Arguments:
        finger_motor_vel -- {'index': [], 'middle': [], ...}
        """
        for finger_name in finger_motor_vel:
            for i in range(len(self.finger_motors[finger_name])):
                if finger_motor_vel[finger_name][i] > self.__DXL_VEL_LIMIT:
                    finger_motor_vel[finger_name][i] = self.__DXL_VEL_LIMIT
                if finger_motor_vel[finger_name][i] < -self.__DXL_VEL_LIMIT:
                    finger_motor_vel[finger_name][i] = -self.__DXL_VEL_LIMIT
                motor_vel = int(
                    finger_motor_vel[finger_name][i] / self.__DXL_VEL_UNIT)
                goal_vel = [
                    dynamixel_sdk.DXL_LOBYTE(
                        dynamixel_sdk.DXL_LOWORD(motor_vel)),
                    dynamixel_sdk.DXL_HIBYTE(
                        dynamixel_sdk.DXL_LOWORD(motor_vel)),
                    dynamixel_sdk.DXL_LOBYTE(
                        dynamixel_sdk.DXL_HIWORD(motor_vel)),
                    dynamixel_sdk.DXL_HIBYTE(
                        dynamixel_sdk.DXL_HIWORD(motor_vel))]
                self.__dxl_groupSyncWrite.addParam(
                    self.finger_motors[finger_name][i], goal_vel)

        dxl_comm_result = self.__dxl_groupSyncWrite.txPacket()
        if dxl_comm_result != dynamixel_sdk.COMM_SUCCESS:
            return False
        self.__dxl_groupSyncWrite.clearParam()
        return True

    def send_motor_torque_cmd_bulk(self, finger_motor_torque):
        """
        Keyword Arguments:
        finger_motor_torque -- {'index': [], 'middle': [], ...}
        """
        for finger_name in finger_motor_torque:
            for i in range(len(self.finger_motors[finger_name])):
                if finger_motor_torque[finger_name][i] > self.__DXL_CURRENT_LIMIT:
                    finger_motor_torque[finger_name][i] = self.__DXL_CURRENT_LIMIT
                if finger_motor_torque[finger_name][i] < - self.__DXL_CURRENT_LIMIT:
                    finger_motor_torque[finger_name][i] = - \
                        self.__DXL_CURRENT_LIMIT
                motor_torque = int(
                    finger_motor_torque[finger_name][i] / self.__DXL_CURRENT_UNIT)
                goal_torque = [dynamixel_sdk.DXL_LOBYTE(dynamixel_sdk.DXL_LOWORD(motor_torque)),
                               dynamixel_sdk.DXL_HIBYTE(dynamixel_sdk.DXL_HIWORD(motor_torque))]
                self.__dxl_groupBulkWrite.addParam(
                    self.finger_motors[finger_name][i], self.__DXL_ADDR_GOAL_CURRENT, 2, goal_torque)

        dxl_comm_result = self.__dxl_groupBulkWrite.txPacket()
        if dxl_comm_result != dynamixel_sdk.COMM_SUCCESS:
            return False
        self.__dxl_groupBulkWrite.clearParam()
        return True