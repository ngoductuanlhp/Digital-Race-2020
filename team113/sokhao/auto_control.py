#!/usr/bin/env python
from __future__ import division
import rospy
# from sensor_msgs.msg import Image
from sensor_msgs.msg import Joy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32
from std_msgs.msg import Char
from std_msgs.msg import Bool
from std_msgs.msg import Int8
# from cv_bridge import CvBridge, CvBridgeError
import cv2
import rospkg 
import math

import pickle
from collections import Counter 

import tensorflow
from tensorflow import keras
from tensorflow.keras import backend
from keras.models import load_model
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
# import cvlib

import numpy as np
import logging, os
import time

from img_processing import ImageProcessing
import predefine_param as pr
import submodules.mycf as cf

import csv

rospack = rospkg.RosPack()
path = rospack.get_path('team113')

error_arr = np.zeros(5)

t = time.time()


countW = 0
keyId = 0

# steer_model = tensorflow.keras.models.load_model(path + '/weights/QNetModel.h5')

elapsed_times = []

def isNaN(num):
    return num != num

class Agent:
    def __init__(self, imgProcessing):
        self.imgProcess = imgProcessing
        self.steer = Float32()
        self.speed = Float32()
        self.steerCam = Float32()

        self.steerPID = 0
        self.steerCNN = 0

        self.count = 0

        self.sign_count = 0;
        self.sign_distance = np.zeros(2)
        self.sign_t = 0

        self.countSign = 0
        self.prevSign = 0
        self.sign_flag = False
        self.detect_sign_count = 0

        self.force_turn_flag = False
        self.total_t = 0
        self.turn_t = 0

        self.distance_obj = -1
        self.prev_t_obj = 0
        self.avoid_obj_flag = 0
        self.prev_pos_obj = None

        self.prev_detect_obj_flag = False
        self.prev_detect_obj_sign = 0
        self.prev_detect_obj_count = 0
        self.detect_obj_flag = False
        self.detect_obj_sign = 0
        self.detect_obj_count = 0
        self.detect_obj_count_4ever = 0

        self.count_increase_speed = 0

        self.flag_vodich = False

        self.pid_arr = [160, 160, 160, 160]
        self.pid_mean = 160

        self.increase_speed_flag = 0

        self.counter_delay = 0

        self.exceed_speed = 0

        self.count_car = 0

        self.raw_steer_pub = rospy.Publisher(pr.new_steer_tp, Float32, queue_size = 1)
        self.raw_speed_pub = rospy.Publisher(pr.new_speed_tp, Float32, queue_size = 1)
        self.raw_steer_cam_pub = rospy.Publisher(pr.new_steercam_tp, Float32, queue_size = 1)
        # test FPS
        self.timestamp = time.time()


    def auto_control(self, num_lane_left, num_lane_right, sign_turn, abnormal, rgb_image, depth_image):
        if cf.manual_control:
            return
        if self.force_turn_flag == True:
            self.detect_obj_flag = False
            self.detect_obj_sign = 0
            self.detect_obj_count = 0
            self.detect_obj_count_4ever = 0
            self.prev_detect_obj_count = 0
            self.prev_detect_obj_flag = False
            self.prev_detect_obj_sign = 0

        '''If already avoid object'''
        if self.detect_obj_flag == True:
            self.detect_obj_count_4ever = self.detect_obj_count_4ever + 1
            self.count_increase_speed = self.count_increase_speed + 1
            if self.count_increase_speed >= 4:
                self.count_increase_speed = 0
                self.increase_speed_flag = 1
            obj_pos_left, obj_pos_right, obj_dis_left, obj_dis_right = self.imgProcess.getDistanceObj(depth_image, rgb_image, self.detect_obj_sign)
            diff = obj_pos_left - obj_pos_right 
            diff_dis = obj_dis_left - obj_dis_right
            # print("Distance from obj: %.2f %.2f" % (obj_distance_left, obj_distance_right))
            # print("Difference: %.2f" % diff)
            if diff > 3.5:
                self.detect_obj_sign = -1
                self.detect_obj_count = 0
            elif diff < -3.5:
                self.detect_obj_sign = 1
                self.detect_obj_count = 0


            ''' CAR ORDERRRRRRRRR'''
            if pr.naive == True:
                if self.count_car in pr.carLeft:
                    self.detect_obj_sign = - 1
                elif self.count_car in pr.carRight:
                    self.detect_obj_sign = 1

            if self.detect_obj_sign == -1:
                print("CAR LEFT")
            elif self.detect_obj_sign == 1:
                print("CAR RIGHT")

            if self.sign_flag == True:
                self.detect_obj_count = self.detect_obj_count + 1

            if abs(diff_dis) > 0 and abs(diff) == 0:
                self.flag_vodich = True
                self.counter_delay = pr.counter_delay
                self.detect_obj_count = 0

            if abs(diff_dis) == 0 and self.flag_vodich == True:
                self.counter_delay = self.counter_delay - 1
            
            if abs(diff_dis) == 0 and self.flag_vodich == False:
                self.detect_obj_count = self.detect_obj_count + 1

            
            # if abs(diff_dis) > 1:
            
            # if abs(diff_dis) >= 3:
            #     self.detect_obj_count = self.detect_obj_count - 1
            if (abs(diff_dis) == 0 and self.counter_delay == 0 and self.flag_vodich == True) or self.detect_obj_count > pr.count_no_obj:
                # print("Turn off ", self.detect_obj_count, self.flag_vodich)
                print("STOP AVOID CAR")
                self.detect_obj_flag = False
                self.detect_obj_sign = 0
                self.detect_obj_count = 0
                self.increase_speed_flag = 0
                self.detect_obj_count_4ever = 0
                self.counter_delay = 0
                self.flag_vodich = False
                self.count_car = self.count_car + 1

        # '''If detect object by cascade'''
        elif self.prev_detect_obj_flag == True:
            
            obj_pos_left, obj_pos_right, obj_dis_left, obj_dis_right = self.imgProcess.getDistanceObj(depth_image, rgb_image, self.detect_obj_sign)
            # print("Distance from obj: %.2f %.2f" % (obj_distance_left, obj_distance_right))
            diff = obj_dis_left - obj_dis_right
            # print("Difference: %.2f" % diff)

            if abs(diff) < 1:
                self.prev_detect_obj_count = self.prev_detect_obj_count + 1

            if self.prev_detect_obj_count == 10:
                self.prev_detect_obj_count = 0
                self.prev_detect_obj_flag = False
                self.prev_detect_obj_sign = 0
            else:
                self.drive(0, num_lane_left, num_lane_right, self.prev_detect_obj_sign, self.prevSign)

        '''If detect sign and detect intersection'''
        if self.force_turn_flag == True:
            time_interval = time.time() - self.turn_t
            # print("Total time: %.4f %d %d" % (time_interval, num_lane_left, num_lane_right))
            if (self.prevSign == -1 and num_lane_left >= 30 and time_interval >= pr.interval_turn1) or time_interval > pr.interval_turn2:
                self.force_turn_flag = False
                self.prevSign = 0
                print("FORWARD")
                self.steer.data = 0
                self.raw_steer_pub.publish(self.steer)
                self.speed.data = 30
                self.raw_speed_pub.publish(self.speed)
                
                # self.drive(0, num_lane_left, num_lane_right)
            elif (self.prevSign == 1 and num_lane_right >= 30 and time_interval >= 1.2) or time_interval > 1.7:
                # self.drive(0, num_lane_left, num_lane_right)
                self.force_turn_flag = False
                self.prevSign = 0
                print("FORWARD")
                self.steer.data = 0
                self.raw_steer_pub.publish(self.steer)
                self.speed.data = 30
                self.raw_speed_pub.publish(self.speed)
            return

        '''If just detect sign'''
        if self.sign_flag == True:
            self.detect_sign_count = self.detect_sign_count + 1
            
            if self.prevSign == -1 and num_lane_left  <= 5 and self.detect_obj_flag != True:
                print("TURN LEFT")
                self.speed.data = 10
                self.raw_speed_pub.publish(self.speed)
                self.steer.data = -70
                self.raw_steer_pub.publish(self.steer)
                self.turn_t = time.time()
                self.sign_flag = False
                self.detect_sign_count = 0
                self.force_turn_flag = True
                
            elif self.prevSign == 1 and num_lane_right <= 5 and self.detect_obj_flag != True:
                print("TURN RIGHT")
                self.speed.data = 10
                self.raw_speed_pub.publish(self.speed)
                self.steer.data = 70
                self.raw_steer_pub.publish(self.steer)
                self.turn_t = time.time()
                self.sign_flag = False
                self.detect_sign_count = 0
                self.force_turn_flag = True
                
            else:
                self.drive(0, 0, 0, self.detect_obj_sign, self.prevSign)

            if self.detect_sign_count == 30:
                self.detect_sign_count = 0
                self.prevSign = 0
                self.sign_flag = False
            return


        '''Normal case'''
        if sign_turn == 0:
            if cf.use_qnet or abnormal:
                self.drive(1)
            else:
                self.drive(0, num_lane_left, num_lane_right, self.detect_obj_sign, self.prevSign)
            return

        '''Detect new sign'''
        self.sign_flag = True
        self.prevSign = sign_turn
        self.detect_sign_count = 0
        if sign_turn == -1:
            print("Detect:\tLEFT SIGN")
        elif sign_turn == 1:
            print("Detect:\tRIGHT SIGN")
        if self.speed.data >= 50:
            self.speed.data = 10
        elif self.speed.data >= 40:
            self.speed.data = 20
        self.raw_speed_pub.publish(self.speed)
        return


    def PID_control(self, error_x, error_y):
        global error_arr, t

        if error_x < 0:
            error = -np.arctan2(error_x, error_y)
        else:
            error = np.arctan2(error_x, error_y)

        p = 43
        i = 1
        d = 2
        
        error_arr[1:] = error_arr[0:-1]
        error_arr[0] = error
        P = error*p
        delta_t = time.time() - t
        t = time.time()
        D = (error-error_arr[1])/delta_t*d
        I = np.sum(error_arr)*delta_t*i
        angle = P + I + D
        if isNaN(angle):
            angle = 0
        if abs(angle) > 60:
            angle = np.sign(angle) * 60
        if error_x > 0:
            return int(angle)
        else:
            return -int(angle)

    def drive(self, chooseSteer, num_left = 0, num_right = 0, obj_flag = 0, turn_flag = 0):
        steerAngle = self.steerPID if chooseSteer == 0 else self.steerCNN
        if isNaN(steerAngle):
            steerAngle = 0
        speed = 0
        offset = abs(steerAngle) // 3
        if offset == 0:
            speed = 55
        elif offset <= 2:
            speed = 50
        elif offset <= 4:
            speed = 45
        else:
            speed = 40

        if (turn_flag != 0 or obj_flag != 0) and chooseSteer != 1:
            if self.increase_speed_flag != 1:
                speed = 10

        if speed == 55:
            self.exceed_speed = self.exceed_speed + 1
            if self.exceed_speed >= 5:
                speed = 50
                self.exceed_speed = 0

        self.speed.data = speed
        self.steer.data = steerAngle

        self.raw_speed_pub.publish(self.speed)
        self.raw_steer_pub.publish(self.steer)
        if obj_flag == 0 and turn_flag == 0 and chooseSteer != 1:
            if num_left < 40:
                if self.steerCam.data == 0:
                    self.steerCam.data = -10
                    self.raw_steer_cam_pub.publish(self.steerCam)
            elif num_right < 40:
                if self.steerCam.data == 0:
                    self.steerCam.data = 10
                    self.raw_steer_cam_pub.publish(self.steerCam)
            else:
                if self.steerCam.data != 0:
                    self.steerCam.data = 0
                    self.raw_steer_cam_pub.publish(self.steerCam)
        else:
            if self.steerCam.data != 0:
                self.steerCam.data = 0
                self.raw_steer_cam_pub.publish(self.steerCam)
