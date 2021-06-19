from __future__ import division
import numpy as np
##########################################
# This is a module for predefined parameter
# WIDTH = 320
# HEIGHT = 240
# KERNEL1
# KERNEL2
# ROI_VERTICES
# CANNY MIN_VAL MAX_VAL
# MIN CONTOURS LEN
# CENTER_SHIFT
# Add more parameters....................
#########################################
image_width = 320
image_height = 240

kernel1 = np.ones((5,5),np.uint8)
kernel2 = np.ones((1,1),np.uint8)
kernel3 = np.ones((9,9),np.uint8)
kernel4 = np.ones((3,3),np.uint8)

vertices = np.array([[0,240],[0,120],[320,120],[320,240],], np.int32)
eliminate_roi_big = np.array([[70,240],[130,120],[190,120],[250,240],], np.int32)
eliminate_roi_turn_left = np.array([[140,240],[160,40],[320,40],[320,240],], np.int32)
eliminate_roi_turn_right = np.array([[0,240],[0,40],[160,40],[180,240],], np.int32)
eliminate_roi_obj = np.array([[130,240],[150,80],[170,80],[190,240],], np.int32)

canny_min_val = 240
canny_max_val = 250
min_contours_len = 50
center_shift = 3

max_point_view_intersection = 14

stick_lane_offset = 30

old_steer_tp = "team113_steerAngle"
old_topic2 = "team113_speed"
new_steer_tp = "team113/set_angle"
new_speed_tp = "team113/set_speed"
new_steercam_tp = "team113/set_camera_angle"

keyboard_tp = "keyboard_pressed"

rgb_transfer_tp = "team113/rgb_img"
depth_transfer_tp = "team113/depth_img"

flag_turn_tp = "flag_turn"
flag_object_tp = "flag_object"

carLeft = []
carRight = [3]

''' Parameter NE XE '''
# De ne xe ko gat
counter_delay = 7 
# tang khi xe dai, giam khi gap nga tu ngay sau ne xe 
count_no_obj = 26

''' NHO TAT GUI NHA '''
gui = True
isLog = False

naive = False

qnet = False

manual = False

'''Scale detect route'''
scale_route = 0.04

''' SMOOTH PID'''
smooth1 = 7
smooth2 = 4
smooth3 = 2

'''INTERVAL TIME'''
interval_turn1 = 1.2
interval_turn2 = 1.7

''' BAM LANE WIDTH'''
bam1 = 28
bam2 = 23
bam3 = 17
bam4 = 13

blue_mask = None
red_mask = None

