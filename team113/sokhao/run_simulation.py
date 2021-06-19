#!/usr/bin/env python3
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
import threading

import numpy as np
import logging, os
import time

from img_processing import ImageProcessing
from auto_control import Agent
import predefine_param as pr

import submodules.mycf as cf

import threading

rgb_lock = threading.Lock()
depth_lock = threading.Lock()


imgProcessing = ImageProcessing()
agent = Agent(imgProcessing)

logData = {'img': [],'steer': [], 'lane': [], 'sign': [], 'obj': []}
keyId = 0
timeRecord = time.time()



def isNaN(num):
    return num != num

def rgb_callback(data):
    global rgb_lock, timeRecord
    try:
        np_arr = np.fromstring(data.data, np.uint8)
        rgb_lock.acquire()
        cf.rgb_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        rgb_lock.release()
        cf.sync_rgb_image = True
    except:
        print("ERROR")

def depth_callback(data):
    global depth_lock
    try:
        np_arr = np.fromstring(data.data, np.uint8)
        depth_lock.acquire()
        cf.depth_image = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
        depth_lock.release()
        cf.sync_depth_image = True
    except:
        print("ERROR")

def joy_callback(joy):
    global agent
    try:
        speed = 0
        # steer = -40 * joy.axes[0]
        # if joy.buttons[0] == 1:
        #     speed = 60
        # elif joy.axes[4] > 0:
        #     speed = 80 * joy.axes[4]

        # if joy.buttons[1] == 1:
        #     cf.isRecord = not cf.isRecord
        steer = -60 * joy.axes[3]
        speed = 55 * joy.axes[1]
        agent.speed.data = speed
        agent.steer.data = steer
        agent.raw_speed_pub.publish(agent.speed)
        agent.raw_steer_pub.publish(agent.steer)
        rospy.sleep(0.1)
    except:
        print("ERROR JOY")

def listenner():
    rospy.Subscriber("/team113/camera/rgb/compressed",
                        CompressedImage, rgb_callback,
                        queue_size = 1, buff_size=2**24)

    rospy.Subscriber("/team113/camera/depth/compressed", 
                        CompressedImage, depth_callback, 
                        queue_size = 1, buff_size=2**24)
    if cf.manual_control:
        rospy.Subscriber("/joy", Joy, joy_callback, 
                        queue_size = 1, buff_size=2**12)



def main_process():
    global agent, logData, keyId, timeRecord
    st = time.time()
    while not rospy.is_shutdown():
        if cf.sync_depth_image and cf.sync_rgb_image:
            # print("FPS:", 1/(time.time() - st))
            # st = time.time()
            cf.sync_depth_image = False
            cf.sync_rgb_image   = False
            
            ''' Synchronize images '''
            rgb_lock.acquire()
            rgb_image = cf.rgb_image.copy()
            rgb_lock.release()

            depth_lock.acquire()
            depth_image = cf.depth_image.copy()
            depth_lock.release()

            cf.debug_image = rgb_image.copy()

            gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV) 

            cf.snow_predict = agent.imgProcess.snowDetectionRF(rgb_image)

            cf.sign_turn, cf.sign_distance = agent.imgProcess.signDetectionRF(rgb_image, depth_image, hsv)

            cf.obj_predict, cf.pos_obj, cf.dis_obj = agent.imgProcess.objectDetectionCascade(rgb_image, gray, depth_image, agent.pid_mean)
            
            ''' Cascade first detect'''
            if cf.obj_predict > 0:
                agent.prev_detect_obj_flag = True
                agent.prev_detect_obj_sign = cf.pos_obj
                if cf.dis_obj < 145:
                    agent.detect_obj_flag = True
                    agent.detect_obj_count = 0
                    agent.prev_detect_obj_flag = False
                    agent.prev_detect_obj_sign = 0
                    agent.prev_detect_obj_count = 0

            cf.center_x, cf.center_y, cf.num_lane_left, cf.num_lane_right = agent.imgProcess.detectLanes(gray, hsv, agent.detect_obj_sign , agent.prevSign)
            if isNaN(cf.center_x):
                cf.center_x = 0
            if isNaN(cf.center_y) or cf.center_y > 120:
                cf.center_y = 120

            agent.steerCNN = 0
            agent.steerPID = agent.PID_control(cf.center_x - 160, cf.center_y)
            if cf.snow_predict == 1 or abs(cf.center_x - 160) > 100: 
                cf.abnormal = 1
                agent.steerCNN = agent.imgProcess.steerDetectionCNN(rgb_image)
            else:
                cf.abnormal = 0

            if cf.abnormal != 1:
                agent.pid_arr[1:3] = agent.pid_arr[0:2]
                agent.pid_arr[0] = cf.center_x
                agent.pid_mean = np.mean(agent.pid_arr)
            
            agent.auto_control(cf.num_lane_left, cf.num_lane_right, cf.sign_turn, cf.abnormal, rgb_image, depth_image)
            
            if cf.isRecord and agent.speed.data > 0 and time.time() - timeRecord >= 0.2:
                timeRecord = time.time()
                logData['img'].append(keyId)
                if cf.abnormal == 1:
                    logData['steer'].append(agent.steerCNN)
                else:
                    logData['steer'].append(agent.steerPID)
                logData['lane'].append([cf.num_lane_left, cf.num_lane_right])
                logData['sign'].append(cf.sign_turn)
                logData['obj'].append(cf.obj_predict)
                cv2.imwrite(saved_path, rgb_image)
                print("Record ", keyId)
                keyId += 1
            
            if cf.show_gui:
                if not cf.manual_control:
                    cv2.imshow("Processed Img", cf.debug_image)
                else:
                    cv2.imshow("RGB", rgb_image)
                    cv2.imshow("Depth", depth_image)
                # cv2.imwrite("/home/tuan/img12.jpg",depth_image)
                # cv2.imshow("Depth", depth_image)
                # if not pr.blue_mask is None:
                #     cv2.imshow("Blue", pr.blue_mask)
                # if not pr.red_mask is None:
                #     cv2.imshow("Red", pr.red_mask)
                cv2.waitKey(1)

def main():
    rospy.init_node('run_simulation_node', anonymous=True)
    rospy.loginfo("Start run_simulation_node")

    listenner()

    main_proc_thread = threading.Thread(name="main_proc", target=main_process)
    main_proc_thread.start()

    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        # df = pd.DataFrame(logData) 
        # saving the dataframe 
        pass

if __name__ == '__main__':
    main()