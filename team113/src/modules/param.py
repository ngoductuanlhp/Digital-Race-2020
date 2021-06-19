import numpy as np

import enum
import cv2

'''##################### ENUM & SYSTEM STATE #####################'''
class SysStates(enum.Enum):
    Stop        = 0
    Ready       = 1
    Running     = 2

class RunStates(enum.Enum):
    Normal      = 0
    HardTurn    = 1
    SwitchLane  = 2

sys_state = SysStates.Stop
run_state = RunStates.Normal

car_base_speed = 22



'''##################### IMPORTANT MACROS #####################'''
isGui           = False
logData         = False 
debugInfo       = False

manual_control  = False



'''##################### PERIPHERAL INPUT #####################'''
ss2_status = False
bt1_status = False
bt2_status = False
bt3_status = False
bt4_status = False


stop_car = False
stopped = False
time_nguoi = 7


'''##################### SYNC IMAGE #####################'''
time_process    = 0

'''##################### IMAGE #####################'''
rgb_image       = None
depth_image     = None
debug_img       = None
preprocessed_img= None

''' Sign '''
red_sign, blue_sign = -1, -1
red_location = (-1,-1)
red_shape = (0, 0)


''' Urgent obstacle '''
pause = False


''' Check racing time '''
count_run = 0
race_time = 0


font = cv2.FONT_HERSHEY_SIMPLEX

count_button4 = 0

time_stop_sign = 0
time_nleft_sign = 0
time_nright_sign = 0
time_left_sign = 0
time_right_sign = 0
time_forward_sign = 0

interpolate_flag = 0

stick_lane = -1
hard_turn_val = -1


num_lane_arr = [100]*5

timer_cursor = -1

parking = False
stop_line = False
center_line  = -1

map_num = 1
two_way_lane = False
wrong_lane = False
parking_center = 180
detect_car = False
use_center = 0
has_snow = False

switch_side = 0
outside = False
check_null_stop = False