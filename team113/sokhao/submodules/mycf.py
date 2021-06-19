import numpy as np
import logging, os
import time

show_gui = True
manual_control = False
isRecord = False

use_qnet = False

prevSign = 0

rgb_image = np.zeros((240, 320, 3), np.uint8)
depth_image = np.zeros((240, 320, 3), np.uint8)
debug_image = np.zeros((240, 320, 3), np.uint8)
sync_rgb_image = False
sync_depth_image = False
sync_processing = True

snow_predict = 0
sign_turn, sign_distance = 0, 0
obj_predict, pos_obj, dis_obj = 0, 0, 0

center_x = 180
center_y = 120
num_lane_left = 0
num_lane_right = 0

abnormal = 0

blueMask = None