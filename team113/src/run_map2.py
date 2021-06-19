#!/usr/bin/env python3
''' ROS libs '''
import rospy
import rospkg
from std_msgs.msg import Bool, Char, String, Float32

''' Python libs '''
from collections import Counter
import cv2
import numpy as np
import time
import threading
import math
import traceback

''' Custom libs '''
import modules.param as par
from modules.car import Car
from modules.camera import Camera
from modules.param import SysStates, RunStates
from modules.image_processing import ImageProcessing
from modules.sign_thread import SignThread
from modules.fill_line import *
from modules.sign_timer import SignTimer


from modules.yolo_thread import YoloThread

''' Global variables '''
rospack = rospkg.RosPack()
path    = rospack.get_path('team113')

''' Publishers '''
lcd_pub = rospy.Publisher('/lcd_print', String, queue_size=1)
led_pub = rospy.Publisher('/led_status', Bool, queue_size=1)

''' Important instances '''
car             = Car()
camera          = Camera()
img_proc        = ImageProcessing()
sign_timer      = SignTimer()

signEventStart = threading.Event()
signEventEnd= threading.Event()
signThread = SignThread(signEventStart, signEventEnd)

yoloEventStart = threading.Event()
yoloEventEnd = threading.Event()
yoloThread = YoloThread(yoloEventStart, yoloEventEnd)

''' Log data variables '''
rgb_arr = []
debug_arr = []
yolo_arr = []
keyId   = 0

''' Obstacle: ss2_status == False '''
def get_ss2_status(data):
    global led_pub, lcd_pub
    if par.ss2_status and not data.data:
        par.pause = True
    if not par.ss2_status and data.data:
        par.pause = False
        print('get_ss2_status')
        if par.sys_state == SysStates.Ready:
            init_and_start()
            print("CAR RUN")
            lcd_pub.publish("00:1:RUNNING         ")
            par.sys_state = SysStates.Running
    par.ss2_status = data.data

''' Top button '''
def get_bt1_status(data):
    par.bt1_status = data.data

def get_bt2_status(data):
    global lcd_pub, car
    if not par.bt2_status and data.data:
        if par.sys_state != SysStates.Stop:
            par.sys_state = SysStates.Stop
            stop_and_reset()
            print("Stop car")
            lcd_pub.publish("00:1:STOPPPPPPPPP")
            saveImg()
            
    par.bt2_status = data.data

def get_bt3_status(data):
    global lcd_pub, car
    if not par.bt3_status and data.data:
        if par.sys_state != SysStates.Running:
            if not par.pause:
                init_and_start()
                print("CAR RUN")
                lcd_pub.publish("00:1:RUNNING    ")
                par.sys_state = SysStates.Running
            else:
                print("CAR READY")
                lcd_pub.publish("00:1:READY      ")
                par.sys_state = SysStates.Ready
    par.bt3_status = data.data

def get_bt4_status(data):
    global lcd_pub
    if not par.bt4_status and data.data:
        par.time_nguoi += 1
        if par.count_button4 == 0:
            lcd_pub.publish("00:3:TEST BUTTON")
        else:
            lcd_pub.publish("00:3:BUTTON TEST")
    par.bt4_status = data.data
    par.count_button4 = (par.count_button4 + 1) % 2

    
def get_angle(data):
    par.raw_angle = int(data.data)

def listener():
    rospy.loginfo("Create callback threads")
    rospy.Subscriber("/ss2_status", Bool, get_ss2_status, queue_size=1)
    rospy.Subscriber("/bt1_status", Bool, get_bt1_status, queue_size=1)
    rospy.Subscriber("/bt2_status", Bool, get_bt2_status, queue_size=1)
    rospy.Subscriber("/bt3_status", Bool, get_bt3_status, queue_size=1)
    rospy.Subscriber("/bt4_status", Bool, get_bt4_status, queue_size=1)

def saveImg():
    global debug_arr, rgb_arr, new_arr, yolo_arr

    if not par.logData:
        print("No logData, skip save images")
        return
    
    for keyId, img in debug_arr:
        saved_path = '/home/ml4u/dira_19_12/debug/img' + str(keyId) + '.png'
        cv2.imwrite(saved_path, img)
        debug_arr = []
    for keyId, img in rgb_arr:
        saved_path = '/home/ml4u/dira_19_12/rgb/img' + str(keyId) + '.png'
        cv2.imwrite(saved_path, img)
        rgb_arr = []
    print("Done save images")

def check_time_sign(t_check, max_time=1, min_time=0):
    t_now = time.time()
    if t_now - t_check >= max_time:
        return False
    if t_now - t_check <= min_time:
        return False
    return True

def init_and_start():
    global car, sign_timer
    par.run_state = RunStates.Normal
    par.stick_lane = -1
    car.base_speed = par.car_base_speed
    car.init_count = 0
    par.timer_cursor = -1
    par.parking = False
    par.stop_line = False
    par.center_line = -1
    par.map_num = 1
    par.switch_side = 0
    par.wrong_lane = False
    sign_timer.sign_array = sign_timer.sign_array_0
    par.race_time = time.time()
    par.switch_side = 0
    par.outside = False
    par.check_null_stop = False
    par.interpolate_flag = 0
    par.stop_car = False
    par.stopped = False

def stop_and_reset(stop_sign = False):
    global car
    print('stop_and_reset')
    car.setSteer(0)
    car.brake()
    if stop_sign:
        par.count_run += 1
        interval = time.time() - par.race_time
        print("######################################################################################")
        print("The %d race, racing time = %.4f" % (par.count_run,interval))
        print("######################################################################################")
    rospy.sleep(6)

def parking():
    global img_proc, keyId
    t_now = time.time()
    if t_now - par.time_stop_sign < 1:
        par.stop_line, par.center_line = img_proc.detectStopLine(par.bev_image)
        if par.parking == False:
            if par.stop_line:
                par.parking = True
        elif par.parking == True:
            if img_proc.timeToStop(par.preprocessed_img):
                print('stoppp',keyId)
                par.parking = False
                return True
    return False

def updateTimeSign():
    global sign_timer
    check = sign_timer.checkValidSign(par.blue_sign, par.red_sign)
    if not check:
        return

    t_now = time.time()
    if par.red_sign == 3:
        par.time_stop_sign = t_now
    elif par.red_sign == 4:
        par.time_nleft_sign = t_now
    elif par.red_sign == 5:
        par.time_nright_sign = t_now

    if par.blue_sign == 1:
        par.time_left_sign = t_now
    elif par.blue_sign == 0:
        par.time_forward_sign = t_now

    sign_timer.update(par.blue_sign, par.red_sign, t_now)
    car.updateSpeed(par.timer_cursor) 

def auto_control(main_center, sub_center, num_sub_lane, num_main_lane, main_lane_angle, sub_lane_angle):
    global car, sign_timer

    # NOTE check parking, neu parking thi khong xu ly tiep
    if par.parking:
        if par.center_line != -1:
            steer = car.steerPID(par.center_line, 250)
            par.parking_center = par.center_line
        else:
            steer = car.steerPID(par.parking_center, 250)
        car.setSteer(steer)
        car.setSpeed(16)
        return

    if par.stop_car and not par.stopped:
        par.stopped = True
        car.brake()
        rospy.sleep(par.time_nguoi)
        return


    # NOTE Switch center
    par.use_center == 0
    if main_lane_angle <= 70 and par.timer_cursor in [4, 6] and not par.wrong_lane:
        center_x, center_y = sub_center[0], sub_center[1]
        par.use_center = 1
    else:
        par.use_center = 0
        center_x, center_y = main_center[0], main_center[1]

    # NOTE chuyen state khi dang o state Normal
    if par.run_state == RunStates.Normal:
        if not par.wrong_lane and par.detect_car and par.timer_cursor == 3: # NOTE ne xe
            par.stick_lane = -par.stick_lane
            par.wrong_lane = True
            par.run_state = RunStates.SwitchLane
            par.switch_side = par.stick_lane
        
        if par.wrong_lane and not par.detect_car and main_lane_angle > -20 and num_main_lane > 50:  # NOTE khi ne on dinh, thay duong thang
            par.stick_lane = - par.stick_lane
            par.wrong_lane = False
            par.run_state = RunStates.SwitchLane
            par.switch_side = par.stick_lane
            
        if num_main_lane <= 30 and not (check_time_sign(par.time_forward_sign) and not check_time_sign(par.time_nleft_sign)): 
            par.run_state = RunStates.HardTurn
            par.hard_turn_val = par.stick_lane
            if par.timer_cursor == 0:
                par.stick_lane = 1
                    
    # NOTE chuyen state khi dang o hard turn
    elif par.run_state == RunStates.HardTurn:
        if not par.mid_vec and num_sub_lane >= 20 and num_main_lane >= 70:
            par.run_state = RunStates.Normal
    
    elif par.run_state == RunStates.SwitchLane:
        if not par.detect_car and num_main_lane >= 70:  
            par.run_state = RunStates.Normal
            par.switch_side = 0

    # NOTE dieu khien cua state
    if par.run_state == RunStates.Normal:
        if center_x == 0 and center_y == 0:
            steer = 0
        elif center_x <= 0:
            steer = -60
        elif center_x >= 360:
            steer = 60
        else:
            steer = car.steerPID(center_x, center_y)
        car.setSteer(steer)

        offset = abs((abs(steer) - 20)) // 10 
        if offset > 0:
            car.setSpeed(car.base_speed - offset)
        else:
            car.setSpeed(car.base_speed)

    elif par.run_state == RunStates.HardTurn:
        car.setSteer(60 * par.hard_turn_val)
        car.setSpeed(18)
    
    elif par.run_state == RunStates.SwitchLane:
        car.setSteer(40 * par.switch_side)
        car.setSpeed(16)

def main_process(lock_rgb, lock_depth):
    global camera, car, keyId
    par.time_process = time.time()

    while not rospy.is_shutdown():
        par.blue_sign, par.red_sign = -1, -1
        par.time_process = time.time()

        if not camera.sync_rgb_image:
            rospy.sleep(0.001)
            continue

        camera.sync_rgb_image = False

        lock_rgb.acquire()
        par.rgb_image = camera.rgb_image.copy()
        lock_rgb.release()

        lock_depth.acquire()
        par.depth_image = camera.depth_image.copy()
        lock_depth.release()

        ''' YOLO Thread '''
        # NOTE for YOLO: input image WxH: 480x288
        yoloThread.img = par.rgb_image[:288,:].copy()
        yoloThread.eventStart.set()

        ''' Sign Thread '''
        # NOTE for SignDetection: input image WxH: 320x180
        signThread.img = par.rgb_image[0:180, 160:480].copy()
        signThread.depth = par.depth_image[0:180, 160:480].copy()
        signThread.eventStart.set()

        
        par.debug_img = cv2.cvtColor(par.rgb_image, cv2.COLOR_RGB2BGR)
        par.bev_image = img_proc.projectBEV(par.rgb_image)

        par.preprocessed_img = img_proc.colorFilterLine(par.bev_image)
        par.mid_vec = img_proc.checkMiddleVector(par.preprocessed_img)

        par.has_snow  =  img_proc.detectSnow(par.preprocessed_img)

        main_center, sub_center,\
            num_main_lane, num_sub_lane,\
            main_lane_angle, sub_lane_angle = img_proc.lane_tracking(
                par.preprocessed_img, lane=par.stick_lane)
        

        ''' Wait for synchronize with TRT threads '''
        signThread.eventEnd.wait()
        signThread.eventEnd.clear()

        yoloThread.eventEnd.wait()
        yoloThread.eventEnd.clear()

        carIdx = (yoloThread.classid == 1)
        carBoxes = yoloThread.boxes[carIdx]
        par.detect_car = False
        if par.map_num == 2:
            for box in carBoxes:
                w = int(box[2] - box[0])
                h = int(box[3] - box[1])
                car_x = int(box[0] + w/2)
                car_y = int(box[1] + h/2)
                top, bot = max(0, car_y-5), min(288,car_y+5)
                left, right = max(0, car_x-5), min(480, car_x+5)
                slide = par.depth_image[top:bot, left:right]
                if np.median(slide) <= 110 and abs(car_x-240) < 130:
                    par.detect_car = True
                break

        if par.sys_state == SysStates.Running:
            updateTimeSign()

            ''' Detect stop sign and stop line '''
            check_parking = parking()
            if check_parking and par.sys_state == SysStates.Running:
                stop_and_reset(True)
                if par.sys_state != SysStates.Stop:
                    if not par.pause:
                        init_and_start()
                        par.sys_state = SysStates.Running
                    else:
                        par.sys_state = SysStates.Ready
                continue
            
            # NOTE Check INTERPOLATE
            if par.interpolate_flag == 0:
                if par.timer_cursor in [2,4,5] and num_main_lane <= 10 and par.run_state == RunStates.Normal:
                    par.interpolate_flag = 1

            else:
                if num_main_lane >= 60:
                    par.interpolate_flag = 0

            if par.interpolate_flag == 1:
                sign_timer.update_interpolate(time.time())
                car.updateSpeed(par.timer_cursor)

                interpolate_img = fill_lane(par.rgb_image)
                interpolate_img = img_proc.projectBEV(interpolate_img)

                main_center, sub_center,\
                    num_main_lane, num_sub_lane,\
                    main_lane_angle, sub_lane_angle = img_proc.lane_tracking(
                        interpolate_img, interpolate=1)

            # NOTE control car
            auto_control(main_center, sub_center, num_sub_lane,
                         num_main_lane, main_lane_angle, sub_lane_angle)
                         
        ''' Debug img '''
        if par.debugInfo:
            center_x, center_y = main_center[0], main_center[1]
            center_x2, center_y2 = sub_center[0], sub_center[1]
            cv2.circle(par.bev_image, (int(center_x),
                                       int(center_y)), 3, (0, 0, 255), -1)
            cv2.circle(par.bev_image, (int(center_x2),
                                       int(center_y2)), 3, (255, 0, 0), -1)

            cv2.putText(par.debug_img, "Num: "+str(num_sub_lane) + "|" +
                        str(num_main_lane), (20, 40), par.font, 0.6, (0, 0, 255), 2, 1)
            cv2.putText(par.debug_img, "Steer:   "+str(car.steer) + " | speed: " +
                        str(car.speed), (20, 70), par.font, 0.6, (0, 0, 255), 2, 1)
            cv2.putText(par.debug_img, "Sign :   "+str(par.blue_sign) + "|" +
                        str(par.red_sign), (20, 100), par.font, 0.6, (0, 0, 255), 2, 1)
            cv2.putText(par.debug_img, "RawAngle:"+str(par.raw_angle),
                        (20, 130), par.font, 0.6, (0, 0, 255), 2, 1)
            cv2.putText(par.debug_img, "stick_lane:"+str(par.stick_lane),
                        (20, 160), par.font, 0.6, (0, 0, 255), 2, 1)
            cv2.putText(par.debug_img, "Center: "+str(center_x)+' '+str(center_y) +
                        ' '+str(par.use_center), (20, 190), par.font, 0.6, (0, 0, 255), 2, 1)
            cv2.putText(par.debug_img, "Lane angle: "+str(int(sub_lane_angle)) + ' ' +
                        str(int(main_lane_angle)), (20, 240), par.font, 0.6, (0, 0, 255), 2, 1)
            cv2.putText(par.debug_img, "TimerCur:"+str(par.timer_cursor),
                        (20, 270), par.font, 0.6, (0, 0, 255), 2, 1)
            cv2.putText(par.debug_img, str(float(time.time() - par.race_time)), (20, 330),
                        par.font, 0.4, (0, 0, 255), 2, 1)

            if par.detect_car:
                cv2.putText(par.debug_img, "car "+str(par.detect_car),
                            (20, 300), par.font, 0.4, (0, 0, 255), 2, 1)
            if par.interpolate_flag == 1:
                cv2.putText(par.debug_img, "interpo", (80, 300),
                            par.font, 0.4, (0, 0, 255), 2, 1)
            if par.mid_vec:
                cv2.putText(par.debug_img, "midvec", (140, 300),
                            par.font, 0.4, (0, 0, 255), 2, 1)
            if par.parking:
                cv2.putText(par.debug_img, "Parkg", (200, 300),
                            par.font, 0.4, (0, 0, 255), 2, 1)
            if par.stop_line:
                cv2.putText(par.debug_img, "StopL", (240, 300),
                            par.font, 0.4, (0, 0, 255), 2, 1)
                cv2.circle(par.bev_image, (int(par.center_line),
                                           int(250)), 3, (0, 255, 0), -1)
            if par.has_snow:
                cv2.putText(par.debug_img, "Snow", (280, 300),
                            par.font, 0.4, (0, 0, 255), 2, 1)
            if par.outside:
                cv2.putText(par.debug_img, "outside", (320, 300),
                            par.font, 0.4, (0, 0, 255), 2, 1)
            if par.detect_car:
                cv2.putText(par.debug_img, "car "+str(par.detect_car),
                            (20, 300), par.font, 0.4, (0, 0, 255), 2, 1)

        img_debug = np.hstack(
                (par.debug_img, cv2.cvtColor(par.bev_image, cv2.COLOR_RGB2BGR)))
        img_debug = cv2.resize(img_debug, (630, 270))
        
        ''' Save data '''
        if par.logData and car.speed > 0 and par.sys_state == SysStates.Running:
            # debug image
            
            debug_arr.append((keyId, img_debug))
            rgb_arr.append((keyId, par.rgb_image))
            keyId += 1

        ''' Show images '''
        if par.isGui:
            cv2.imshow("Debug", img_debug)
            k = cv2.waitKey(1)
            if k == ord('s') or k == ord('s'):
                par.logData = not par.logData
                print("Change logData: ", par.logData)
            elif k & 0xFF == 27:
                return



'''--------------------------------------------- Main function ---------------------------------------------'''
def main():
    global camera, car, signThread, yoloThread

    rospy.init_node('run_node', anonymous=True)
    rospy.loginfo("START run_node")
    try:
        ''' Create listener threads '''
        listener()

        ''' Inference threads '''
        signThread.start()
        yoloThread.start()
        
        rospy.sleep(0.05)

        lock_rgb = threading.Lock()
        lock_depth = threading.Lock()

        depth_thread = threading.Thread(
            name="depth", target=camera.getDepth, args=[lock_depth])
        depth_thread.start()
        rospy.sleep(0.05)

        rgb_thread = threading.Thread(
            name="rgb", target=camera.getRGB, args=[lock_rgb])
        rgb_thread.start()
    
        ''' Main thread '''
        main_process(lock_rgb, lock_depth)
    except:
        print("BUGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG")
        car.brake()
        traceback.print_exc()
    finally:
        car.brake()
        camera.shutdown()
        rgb_thread.join()
        depth_thread.join()
        signThread.stop()
        yoloThread.stop()
        cv2.destroyAllWindows()
        rospy.loginfo("STOP run_node")

if __name__ == '__main__':
    main()