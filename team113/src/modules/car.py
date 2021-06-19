import rospy
from std_msgs.msg import Float32
from sensor_msgs.msg import Joy
import modules.param as par
from modules.param import RunStates, SysStates
import time
import numpy as np
import math

class Car:
    def __init__(self):
        ''' Manual control '''
        self.base_speed = par.car_base_speed
        self.init_speed = 16
        self.init_count = 0

        self.error_arr = np.zeros(5)
        self.prev_normal_steer = np.zeros(5)
        self.time_pid = time.time()

        self.steer = 0

        self.speed = 0

        self.steer_pub = rospy.Publisher("/set_angle", Float32, queue_size=1)
        self.speed_pub = rospy.Publisher("/set_speed", Float32, queue_size=1)

    
    def updateSpeed(self, cursor):
        if cursor == 4:
            self.base_speed = 18
        elif cursor == 5:
            self.base_speed = 20
        elif cursor == 1:
            self.base_speed = 18

    def setSteer(self, steer):
        ''' From left to right: -60 to 60 '''
        if par.sys_state != SysStates.Running:
            return

        if par.pause:
            self.steer = 0
            self.steer_pub.publish(0)
        else:
            self.steer = steer
            self.steer_pub.publish(steer)

    def setSpeed(self, speed):
        if par.pause:
            self.speed_pub.publish(0)
            self.speed = 0
            return

        if par.sys_state != SysStates.Running:
            return


        # NOTE For init run
        if self.init_count <= self.base_speed - self.init_speed:
            speed = self.init_speed + self.init_count
            self.init_count += 1
        
        self.speed = speed
        self.speed_pub.publish(speed)

    def brake(self):
        self.steer = 0
        self.steer_pub.publish(0)
        self.speed = 0
        self.speed_pub.publish(0)


    def steerPID(self, center_x, center_y):
        steer = 0
        if center_x == -1:
            steer = 0
        else:
            center_x_smooth = 5 * (center_x // 5)
            error = (center_x_smooth - 180)
            
            p = 0.28
            i = 0.005
            d = 0.005
            ''' Calculate P '''
            self.error_arr[1:] = self.error_arr[0:-1]
            self.error_arr[0] = error

            P = error*p
            delta_t = time.time() - self.time_pid
            self.time_pid = time.time()

            ''' Calculate I '''
            I = np.sum(self.error_arr)*delta_t*i

            ''' Calculate D '''
            D = (error-self.error_arr[1])/delta_t*d

            steer = P + I + D
            # print("car steer:", steer, P, np.arctan2(error_x, error_y)*45)
            if math.isnan(steer):
                steer = 0
            if abs(steer) > 60:
                steer = np.sign(steer) * 60
            self.prev_normal_steer[1:5] = self.prev_normal_steer[0:4]
            self.prev_normal_steer[0]   = steer
            par.pid_steer = int(steer)
        return int(steer)

    def joy_control(self, joy):
        ''' MOD 0 - DIRECT INPUT '''
        # RB:                   Constant speed
        # y-axis right stick:   Linear speed
        # x-axis left stick:    Linear steer
        # X button:             Toggle record
        offset = 1 if abs(joy.axes[0]) >= 0.5 else 0 
        if joy.buttons[5] == 1:
            speed = 18 + offset
        elif joy.axes[3] > 0:
            speed = mapValue(joy.axes[3], 0, 1, 10, 16) + offset
        else:
            speed = 0
        
        if abs(joy.axes[0]) <= 0.8:
            steer = -70 * joy.axes[0]
        else:
            steer = -60 * np.sign(joy.axes[0])

        if joy.buttons[0] == 1:
            par.logData = not par.logData
            if par.logData == True:
                print("X-button: RECORD")
            else:
                print("X-button: STOP RECORD")

        # print("Speed: %.2f, Steer: %.2f" % (speed, steer))
        return int(speed),int(steer)

def mapValue(val, min_in, max_in, min_out, max_out):
    if val < min_in:
        return min_out
    if val > max_in:
        return max_out
    return min_out + val * (max_out - min_out)/(max_in - min_in)



            

        
        