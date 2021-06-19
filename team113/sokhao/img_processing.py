#!/usr/bin/env python
from __future__ import division

''' ROS libs'''
import rospy
import rospkg

''' Python libs '''
import numpy as np
import logging, os
import time
from collections import Counter

''' OpenCV libs '''
import cv2
# import cvlib


''' ML/DL libs '''
import tensorflow
from tensorflow import keras
from tensorflow.keras import backend
from keras.models import load_model
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import pickle
# import cPickle

''' Customized libs '''
import predefine_param as pr
import submodules.mycf as cf

''' Get path to ROS package '''
rospack = rospkg.RosPack()
path = rospack.get_path('team113')

''' Config for tensorflow-GPU '''
config = tensorflow.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tensorflow.compat.v1.Session(config=config)

''' Log data key '''
keyId_sign = 3851
keyId_obj = 18332
countS = 1

keyId1 = 801
keyId2 = 505

''' Additional functions '''
def isNaN(num):
    return num != num

def roi(img, vertices, inverse = False):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    if inverse is True :
        mask = cv2.bitwise_not(mask)
    masked = cv2.bitwise_and(img, mask)
    return masked

def detect_outlier(data_1,threshold):
    outliers=[]
    #threshold=2
    mean_1 = np.mean(data_1)
    std_1 = np.std(data_1)
    
    
    for y in data_1:
        z_score= (y - mean_1)/ float(std_1) 
        if np.abs(z_score) > threshold:
            outliers.append(y)
    return outliers


class ImageProcessing:
    def __init__(self):
        ''' Sign classification using random forest '''
        with open(path + '/weights/sign_3_1.pkl', 'rb') as fid:
            self.sign_model_rf = pickle.load(fid, encoding="latin1")

        ''' Snow detection using random forest '''
        with open(path + '/weights/snow_detection.pkl', 'rb') as fid:
            self.snow_model_rf = pickle.load(fid, encoding="latin1")

        ''' Object detection using random forest '''
        with open(path + '/weights/object_filter_3_1.pkl', 'rb') as fid:
            self.obj_model_rf = pickle.load(fid, encoding="latin1")

        ''' Steer prediction using CNN QNet '''
        self.steer_model = tensorflow.keras.models.load_model(path + '/weights/QNetModel.h5')

        # '''Yolo object detection'''
        # self.yolo_model = tensorflow.keras.models.load_model(path + '/weights/yolo.h5')
        
        ''' Object detection using cascade '''
        self.car_cascade = cv2.CascadeClassifier(path + '/cascades/cars_final.xml')


        '''Lane width'''
        self.lane_width = 0

        ''' Object coordinates '''
        self.coor_obj_y = 0
        self.coor_obj_h = 0
        self.coor_obj_x = 0
        self.coor_obj_w = 0

        ''' Previous PID '''
        self.prev_pid = 160

        ''' Previous contour'''
        self.prevCon = np.zeros((240,320), np.uint8)

        self.blue_low = np.array([85, 120, 20], dtype=np.uint8)
        self.blue_high = np.array([130, 240, 250], dtype=np.uint8)

        self.red_low     = np.array([155, 120, 20], dtype=np.uint8)
        self.red_high    = np.array([185, 240, 250], dtype=np.uint8)

        ''' warm-up '''
        sign = -1 + 2 * np.array(np.zeros((128,128,2), np.uint8)).reshape(128*128*2).astype(np.float32) / 255.0
        self.sign_model_rf.predict_proba([sign])
        snow = np.array(np.zeros((128,128,3), np.uint8)).reshape(128*128*3).astype(np.float32) / 255.0
        self.snow_model_rf.predict([snow])
        obj = -1 + 2 * np.array(np.zeros((128,128,3), np.uint8)).reshape(128*128*3).astype(np.float32) / 255.0
        self.obj_model_rf.predict([obj])
        img = np.asarray([(-1 + 2 * np.asarray(np.zeros((112,200,3), np.uint8))/255.0)])
        self.steer_model.predict(img)

    def signDetectionRF(self, rgb, depth, hsv):
        global keyId_sign, path

        offset = 8	
        turn = 0
        distance = 0

        img_rgb = rgb[0:140,...]
        img_depth = depth[0:140,...]


        # self.getBlue(rgb)

        lower_blue = np.array([90,60,35])
        upper_blue = np.array([130,255,255])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        # res = cv2.bitwise_and(img_rgb, mask)

        ''' Filter contours'''
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 15 and area < 90:
                x,y,w,h = cv2.boundingRect(cnt)
                if x > 130 and y > 40 and w/h > 0.5 and w/h < 1.5:
                    # print('sign 1')
                    distance = np.mean(img_depth[y + h//2 - offset : y + h//2 + offset,
                                        x + w//2 - offset : x + w//2 + offset])
                    if distance < 160 and distance > 40 and not isNaN(distance):
                        ''' Crop sign image to predict '''
                        crop = img_rgb[y-10:y+h+5,x-5:x+w+10]

                        # print('sign 2')

                        ''' Clear background '''
                        b, g, r = cv2.split(crop)
                        crop_depth = img_depth[y-10:y+h+5,x-5:x+w+10]
                        ret, depth_mask = cv2.threshold(crop_depth, distance + 3, 255, cv2.THRESH_BINARY_INV)
                        b = cv2.bitwise_and(b, depth_mask)
                        g = cv2.bitwise_and(g, depth_mask)
                        r = cv2.bitwise_and(r, depth_mask)

                        # if pr.isLog:
                            # crop_save = cv2.merge((b,g,r))
                            # crop_save = cv2.resize(crop_save, (128, 128))
                            # cv2.imwrite('/home/tuan/catkin_ws/data/sign_2_1/img' + str(keyId_sign) + '.jpg', crop_save)
                            # keyId_sign += 1
                        # if pr.gui:
                        #     cv2.imshow("Sign", crop)

                        ''' Convert to proper data type '''
                        crop = cv2.merge((g,r))
                        crop = cv2.resize(crop, (128, 128))
                        crop = np.array(crop)
                        crop = crop.reshape(128*128*2)
                        crop = -1 + 2 * crop.astype(np.float32) / 255.0
                        
                        ''' Predict probability '''
                        # turn = self.sign_model_rf.predict([crop])
                        predicted_proba = self.sign_model_rf.predict_proba([crop])[0].tolist()
                        max_proba = max(predicted_proba)
                        sign_type = predicted_proba.index(max_proba)
                        # max_pred = np.amax(turn_p)
                        # idx = np.where(turn_p == max_pred)
                        # print("Prob: ", max_proba)
                        
                        if sign_type != 1:
                            if max_proba >= 0.62:
                                cv2.rectangle(cf.debug_image,(x,y),(x+w,y+h),(0,255,255),2)
                                # print("SIGN: ", idx)
                                # print("%d %.2f" %(idx, max_pred))
                                if sign_type == 0:
                                    turn = -1
                                elif sign_type == 2:
                                    turn = 1   
                        else:
                            distance = 0
                            turn = 0     
        return turn, distance

    def snowDetectionRF(self, rgb):
        rgb = rgb[80:240, 70:250]
        crop = np.array(rgb)
        crop = cv2.resize(crop, (128, 128))
        crop = crop.reshape(128*128*3)
        crop = crop.astype(np.float32) / 255.0
        snow = self.snow_model_rf.predict([crop])[0]
        return snow

    def steerDetectionCNN(self, rgb):
        cv_image = rgb[60:240,0:320]
        cv_image = cv2.resize(cv_image,(200,112))
        cv_image = -1 + 2*np.asarray(cv_image)/255.0
        cv_image = np.asarray([cv_image])
        steer = self.steer_model.predict(cv_image) * 40
        return steer

    # def objectDetection(self, rgb):
    #     cv_image = rgb[60:240, 0:320]
    #     cv_image = cv2.resize(cv_image,(128, 128))
    #     cv_image = -1 + 2*np.asarray(cv_image)/255.0
    #     cv_image = np.asarray([cv_image])
    #     objPredict = self.yolo_model.predict(cv_image)
    #     temp = np.asarray(objPredict[0][0][0])
    #     print(temp.shape)
    #     return 1

    def objectDetectionCascade(self, rgb, gray, img_depth, center_x):
        global keyId_obj
        offset = 3
        count_obj = pos_obj = distance = 0
        # med = cv2.medianBlur(gray,25)
        
        # gray = cv2.equalizeHist(gray)
        cars = self.car_cascade.detectMultiScale(gray,
            scaleFactor = 1.04,
            minNeighbors = 5,
            minSize=(20, 20),
            maxSize=(160, 160),
            flags = 0)

        for (x, y, w, h) in cars:
            # cv2.rectangle(img_rgb,(x,y),(x+w,y+h),(0,0,255),2)
            # cv2.imshow("Obj", img_rgb[y:y+h, x:x+w])
            # print("Detect: %d %d %d %d" %(x, y, w, h))
            obj_x = int(x + w/2.0)
            obj_y = int(y + h/2.0)
            # if x > 30 and x < 250 and y > 30 and abs(center_x - obj_x) < 50:
            # if obj_x > 70 and obj_x < 250 and obj_y > 40 and obj_y < 200:
            if obj_x > 40 and obj_x < 240 and obj_y > 40 and obj_y < 200 and abs(obj_x - center_x) < self.lane_width/float(2) -10:
                distance = np.mean(img_depth[y + h//2 - offset : y + h//2 + offset,
                                    x + w//2 - offset : x + w//2 + offset])
                # print("car 1")
                # print("DIs: ", distance)
                # print("Last: ", abs(obj_x - center_x) < self.lane_width/float(2))
                if distance < 190:
                    # cv2.rectangle(rgb,(x,y),(x+w,y+h),(0,0,255),2)
                    crop = rgb[y:y+h, x:x+w]
                    crop = cv2.resize(crop, (128, 128))
                    # print("car 2")
                    # crop_save = cv2.resize(crop, (128, 128))
                    # if pr.isLog:
                    #     cv2.imwrite('/home/tuan/catkin_ws/data/obj_2_1/img' + str(keyId_obj) + '.jpg', crop_save)
                    #     keyId_obj += 1

                    crop = np.array(crop)
                    crop = crop.reshape(128*128*3)
                    crop = -1 + 2 * crop.astype(np.float32) / 255.0
                    obj = self.obj_model_rf.predict([crop])
                    if obj[0] == 1:
                        cv2.rectangle(cf.debug_image,(x,y),(x+w,y+h),(0,0,255),2)
                        if obj_x < 160:
                            pos_obj = -1
                        else:
                            pos_obj = 1
                        # print("Position: ", pos_obj)
                        count_obj = 1
                        self.coor_obj_y = y
                        self.coor_obj_h = h
                        self.coor_obj_x = x
                        self.coor_obj_w = w
                        return count_obj, pos_obj, distance
                    # roi = img_rgb[y:y+h, x:x+w]
                    # cv2.imshow("Obj", roi)
                else:
                    distance = 0
        
        # if pr.gui == True:
        #     cv2.imshow('Object', rgb)
        self.coor_obj_y = 0
        self.coor_obj_h = 0
        self.coor_obj_x = 0
        self.coor_obj_w = 0
        return count_obj, pos_obj, distance

    def filterRoute(self, roi_left, roi_right):
        area_left = roi_left.shape[0] * roi_left.shape[1]
        area_right = roi_right.shape[0] * roi_right.shape[1]
        # cv2.imshow("Left", roi_left)
        # cv2.imshow("Right", roi_right)
        route_low = np.array([0, 0, 0], dtype=np.uint8)
        route_high = np.array([170, 40, 120], dtype=np.uint8)

        gray_low = np.array((0), dtype=np.uint8)
        gray_high = np.array((113), dtype=np.uint8)


        hsv_left = cv2.cvtColor(roi_left, cv2.COLOR_BGR2HSV)
        gray_left = cv2.cvtColor(roi_left, cv2.COLOR_BGR2GRAY)
        # mask_gray_left = cv2.inRange(gray_left, gray_low, gray_high)
        mask_left = cv2.inRange(hsv_left, route_low, route_high)
        # mask1 = cv2.bitwise_and(mask_gray_left, mask_left)
        num_left = len([np.where(mask_left==255)][0][0])


        hsv_right = cv2.cvtColor(roi_right, cv2.COLOR_BGR2HSV)
        gray_right = cv2.cvtColor(roi_right, cv2.COLOR_BGR2GRAY)
        # mask_gray_right = cv2.inRange(gray_right, gray_low, gray_high)
        mask_right = cv2.inRange(hsv_right, route_low, route_high)
        # mask2 = cv2.bitwise_and(mask_gray_right, mask_right)
        num_right = len([np.where(mask_right==255)][0][0])

        # print("Num roi: %d %d" % (num_left, num_right))
        s1 = num_left/float(area_left)
        s2 = num_right/float(area_right)
        diff = abs(s1 - s2) 
        # print("Diff route: ", diff)
        if diff > pr.scale_route:
            if s1 > s2:
                return 0, 5
            return 5,0
        return 0,0

        # if num_left/float(area_left) > num_right/float(area_right):
        #     print("ti le: ", num_left/float(area_left), num_right/float(area_right))
        #     cv2.imshow("Left", roi_left)
        #     cv2.imshow("Right", roi_right)
        #     return 0, 5
        # return 5, 0

    def filterHist(self, depth):
        hist_left = cv2.calcHist(depth[140:240, 60:160],[0],None,[256],[0,256]).flatten()
        hist_right = cv2.calcHist(depth[140:240, 160:260],[0],None,[256],[0,256]).flatten()
        
        count_left = np.sum(hist_left[80:130])
        count_right = np.sum(hist_right[80:130])

        # print("Hist: %d %d" % (count_left, count_right))

        diff =  count_left - count_right

        if diff > 30:
            return 5, 0
        if diff < - 30:
            return 0, 5
        return 0, 0

        # print("Hist count: %d, %d" % (count_left, count_right))

    def getFrontDistance(self, depth):
        offset = int(self.lane_width / float(2))
        distance_left = np.mean(depth[140: 200, 160 - offset -20: 140])
        distance_right = np.mean(depth[140: 200, 180 : 160 + offset + 20])
        return distance_left, distance_right

    def getFrontRoute(self, rgb, flag):
        # hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
        offset = int(self.lane_width / float(4))
        area = 80 * 2 * offset
        roi_center = rgb[140:220, 160 - offset:160 + offset]
        if flag == -1:
            roi_obj = rgb[140:220, 160 - 3 * offset:160 - offset]
        else:
            roi_obj = rgb[140:220, 160 + offset:160 + 3 * offset]

        route_low = np.array([0, 0, 0], dtype=np.uint8)
        route_high = np.array([160, 40, 115], dtype=np.uint8)

        hsv_center = cv2.cvtColor(roi_center, cv2.COLOR_BGR2HSV)
        # gray_center = cv2.cvtColor(roi_left, cv2.COLOR_BGR2GRAY)
        mask_center = cv2.inRange(hsv_center, route_low, route_high)
        num_center = len([np.where(mask_center==255)][0][0])
        

        hsv_obj = cv2.cvtColor(roi_obj, cv2.COLOR_BGR2HSV)
        # gray_obj = cv2.cvtColor(roi_obj, cv2.COLOR_BGR2GRAY)
        mask_obj = cv2.inRange(hsv_obj, route_low, route_high)
        num_obj = len([np.where(mask_obj==255)][0][0])

        diff = abs(num_obj/float(area) - num_center/float(area))

        # print("Diff:", diff)
        if diff <= 0.05:
            return 0, 0
        return 5, 0

    def detectLineObj(self, depth, flag):
        # cv2.imshow("Depth img ", depth)
        # route_line = np.array(depth[210:211, 157:163])[0]
        # route_val = Counter(route_line).most_common(1)[0][0]
        offset = int(self.lane_width) if self.lane_width <= 160 else 160
        if flag == -1:
            line = depth[210:211, 160 - offset: 160]
        else:
            line = depth[210:211, 160: 160 + offset]
        
        route_val = int(np.mean(line))

        line = np.array(line)[0]
        obj_line = [x for x in line if x < route_val - 2]
        # print("Array", obj_line)
        num = len(obj_line)
        # print("DEBUG: width: %d, mean: %d, num: %d" % (self.lane_width, route_val, num))
        if num < 4:
            return 0,0
        return 5,0

    def getDistanceObj(self, depth, rgb, flag):
        distance_left, distance_right = self.detectLineObj(depth, flag)
        if self.coor_obj_h == 0 and self.coor_obj_y == 0 and self.coor_obj_w == 0 and self.coor_obj_x == 0:
            # obj_pos_left, obj_pos_right = self.filterHist(depth)
            #offset = int(self.lane_width / float(2))
            # distance_left = np.mean(depth[140: 200, 160 - offset: 160])
            # distance_right = np.mean(depth[140: 200, 160 : 160 + offset])
            # cv2.imshow("Left", depth[140: 200, 160 - offset : 160])
            # cv2.imshow("Right", depth[140: 200, 160 : 160 + offset])
            # # distance_left, distance_right = self.getFrontRoute(rgb, flag)
            # print("Distance: %.4f %.4f" % (distance_left, distance_right))
            # distance_left, distance_right = self.getFrontDistance(depth)
            # distance_left, distance_right = self.detectLineObj(depth, flag)
            return 0, 0, distance_left, distance_right

        # distance_left = np.mean(depth[self.coor_obj_y: self.coor_obj_y + self.coor_obj_h, 70 : 140])
        # distance_right = np.mean(depth[self.coor_obj_y: self.coor_obj_y + self.coor_obj_h, 180 : 250])
        obj_x = self.coor_obj_x + self.coor_obj_w/2
        if obj_x < 100:
            return 5, 0, distance_left, distance_right
        elif obj_x > 220:
            return 0, 5, distance_left, distance_right
        else:
            # roi_edge = min(int(self.lane_width/float(2)), self.coor_obj_x - 60, 260 - self.coor_obj_x - self.coor_obj_w)
            
            # roi_edge = min(int(self.lane_width/float(2)), self.coor_obj_x, 260 - obj_x)
        
            # print("BUG: ", self.coor_obj_x + self.coor_obj_w, self.coor_obj_x + self.coor_obj_w + roi_edge)
            roi_edge = min(int(self.lane_width/float(2)), self.coor_obj_x - 60, 260 - obj_x)
            if roi_edge <= 0:
                roi_edge = 1
            obj_pos_left, obj_pos_right = self.filterRoute( rgb[self.coor_obj_y + int(self.coor_obj_h * float(3/4)): self.coor_obj_y + self.coor_obj_h,
                                                            int(max(0, self.coor_obj_x - roi_edge)) : self.coor_obj_x],
                                                            rgb[self.coor_obj_y + int(self.coor_obj_h * float(3/4)): self.coor_obj_y + self.coor_obj_h, 
                                                            self.coor_obj_x + self.coor_obj_w : int(min(320, self.coor_obj_x + self.coor_obj_w + roi_edge))])
        return obj_pos_left, obj_pos_right, distance_left, distance_right
    

    def preprocessingImg(self, mask):

        ''' Eliminate noise 1st time '''
        contours, hie = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        final_contours = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            rec_area = w * h
            extent = float(area) / rec_area

            if rec_area > 70 and extent < 0.8 and (w > 25 or h > 25) and y < 200:
                final_contours.append(contour)
        mask_denoise = np.zeros((240,320), np.uint8)
        cv2.drawContours(mask_denoise, final_contours, -1, (255, 255, 255), 3)

        
        ''' Detect and smoothen edges '''
        _, threshold_img1 = cv2.threshold(mask_denoise, 60,255,cv2.THRESH_BINARY)
        edges = cv2.Canny(threshold_img1, pr.canny_min_val, pr.canny_max_val)
        #closing_morphology_filter
        closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))
        #erode_morphology_filter
        erosion = cv2.erode(closed_edges, pr.kernel2, iterations = 1)
        ret, threshold_img2 = cv2.threshold(erosion, 64, 255, 0)
        
        ''' Eliminate noise 2nd time '''
        contours, hie = cv2.findContours(threshold_img2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        final_contours = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            rec_area = w * h
            #print("Area", rec_area)
            extent = float(area) / rec_area
            if len(contour) > 25:
                if extent < 0.3:
                    final_contours.append(contour)
                elif rec_area > 400 and y < 160 and y > 50 and (w < 25 or h < 25 or extent < 0.9):
                    final_contours.append(contour)
        processed_img = np.zeros((240,320), np.uint8)
        cv2.drawContours(processed_img, final_contours, -1, (255, 255, 255), 3)
        temp = processed_img.copy()
        processed_img = cv2.bitwise_or(self.prevCon, processed_img)
        self.prevCon = temp
        # if pr.gui:
        #     cv2.imshow("Contour", processed_img)

        #dilation_morphology_filter
        processed_img = cv2.dilate(processed_img, pr.kernel1, iterations = 1)
        #closing_morphology_filter
        processed_img = cv2.morphologyEx(processed_img, cv2.MORPH_CLOSE, pr.kernel1)
        

        if pr.gui:
            cv2.imshow("Contour", processed_img)

        return processed_img

    def detectLeftRight(self, processedImg, obj_flag = 0, turn_flag = 0):
        leftLane, rightLane = [],[]
        #Traverse from bottem center
        skip = 0
        for i in range(239,119,-1):
            row = processedImg[i]
            #denoise
            #print(pr.image_width/2 - pr.center_shift,pr.image_width/2 + pr.center_shift)
            # center = row[pr.image_width//2 - pr.center_shift:pr.image_width//2 + pr.center_shift]
            # if (len(np.where(center == 255)[0]) > 20 or i == 120):
            #     break
            left_part,right_part = row[0:160],row[160:319]
            '''Find left lane'''
            leftLane += [np.where(left_part==255)[0][-1]] if (len(np.where(left_part==255)[0])>0)else [0]
            '''Find right lane'''
            rightLane += [np.where(right_part==255)[0][0]+160] if (len(np.where(right_part==255)[0])>0)else [320]

        if obj_flag != 0:
            # offset_lane = 4 + 5 * int((self.lane_width - 40)/ float(20))
            if self.lane_width > 130:
                offset_lane = pr.bam1
            elif self.lane_width > 100:
                offset_lane = pr.bam2
            elif self.lane_width > 70:
                offset_lane = pr.bam3
            else:
                offset_lane = pr.bam4
        
            if obj_flag == 1 :
                return np.mean(leftLane[::-1][5:20]) + offset_lane, 120, 0, 0
            elif obj_flag == -1 :
                return np.mean(rightLane[::-1][5:20]) - offset_lane, 120, 0, 0


        center = []
        width = []
        sliding_window = []
        sliding_window_res = []
        max_window_num = 20
        window_size = 1
        temp_window_num = 0

        if turn_flag == 0 :
            num_left = len([x for x in leftLane if x != 0])
            num_right = len([x for x in rightLane if x != 320])
        elif turn_flag != 0 :
            num_left = len([x for x in leftLane[40:70] if x != 0])
            num_right = len([x for x in rightLane[40:70] if x != 320])

        for i in range (5,len(leftLane)-window_size,window_size):
            left_window = leftLane[len(leftLane)-(i+window_size):len(leftLane)-i]
            right_window = rightLane[len(rightLane)-(i+window_size):len(rightLane)-i]

            temp = (np.mean(left_window)+np.mean(right_window))//2

            width_temp = np.mean(right_window) - np.mean(left_window)
            if (0.0 in left_window) or  (320.0 in right_window):
                pass
            else:
                temp_window_num = temp_window_num + 1
                sliding_window_res.append(temp)
                sliding_window.append(temp)
                width.append(width_temp)
                if(temp_window_num == max_window_num):
                    break
        
        
        # print("Lane width: ", self.lane_width)
        if len(sliding_window) == 0:
            center_x = 160
            center_y = 120
            return center_x, center_y, num_left, num_right 

        self.lane_width = np.mean(width)

        center_x = Counter(sliding_window).most_common(1)[0][0]
        center_y = len(leftLane)-sliding_window_res.index(Counter(sliding_window).most_common(1)[0][0])*window_size+skip
        self.lane_width = np.mean(width)
        return center_x, center_y, num_left, num_right

    def detectLanes(self, gray, hsv, obj_flag = 0, turn_flag = 0):

        # hls = cv2.cvtColor(rgb, cv2.COLOR_BGR2HLS)

        ''' Color filtering, convert to binary image '''
        gray_mask = cv2.inRange(gray,(127),(255))

        white_low = np.array([0, 0, 180], dtype=np.uint8)
        white_high = np.array([150, 40, 255], dtype=np.uint8)
        white_mask = cv2.inRange(hsv, white_low, white_high)

        shadow_low = np.array([70, 3, 80], dtype=np.uint8)
        shadow_high = np.array([110, 50, 179], dtype=np.uint8)
        shadow_mask = cv2.inRange(hsv, shadow_low, shadow_high)

        border_low = np.array([10, 40, 180], dtype=np.uint8)
        border_high = np.array([30, 70, 255], dtype=np.uint8)
        border_mask = cv2.inRange(hsv, border_low, border_high)

        final_mask = white_mask | shadow_mask | border_mask | gray_mask

        ''' Get region of interest '''
        final_mask = roi(final_mask, [pr.vertices])
        # cv2.imshow("Final mask", final_mask)
        # cv2.imwrite("/home/tuan/1.jpg", final_mask)

        if obj_flag != 0 :
            pass
            #final_mask = roi(final_mask ,[pr.eliminate_roi_obj], True)
        elif turn_flag == -1:
            final_mask = roi(final_mask ,[pr.eliminate_roi_turn_left], True)
        elif turn_flag == 1:
            final_mask = roi(final_mask ,[pr.eliminate_roi_turn_right], True)
        else:
            final_mask = roi(final_mask ,[pr.eliminate_roi_big], True)

        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, pr.kernel2)
        # cv2.imshow("Final mask2", final_mask)
        # cv2.imwrite("/home/tuan/2.jpg", final_mask)

        ''' Detect 2 white lines, denoise, return noise-free binary image '''
        processedImg = self.preprocessingImg(final_mask)

        ''' Get 2 white lines, calculate center of route and left/right lines '''
        center_x, center_y, num_left, num_right = self.detectLeftRight(processedImg, obj_flag, turn_flag)

        ''' Smoothen center point '''
        if center_x > self.prev_pid :
            # print("Giut: ", center_x - self.prev_pid)
            if center_x - self.prev_pid > 50:
                center_x = center_x - pr.smooth1
            elif center_x - self.prev_pid > 40:
                center_x = center_x - pr.smooth2
            elif center_x - self.prev_pid > 30:
                center_x = center_x - pr.smooth3
        elif center_x < self.prev_pid :
            # print("Giut: ", center_x - self.prev_pid)
            if center_x - self.prev_pid < -50:
                center_x = center_x + pr.smooth1
            elif center_x - self.prev_pid < -40:
                center_x = center_x + pr.smooth2
            elif center_x - self.prev_pid < -30:
                center_x = center_x + pr.smooth3
        #center_x = center_x - ((center_x - self.prev_pid)%30)*7
        
        self.prev_pid = center_x
        return center_x, center_y, num_left, num_right

    def getBlue(self, rgb):
        global keyId1, keyId2
        kernel = np.ones((5,5),np.uint8)
        rgb = cv2.resize(rgb, (480,360))
        hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
        blue_mask = cv2.inRange(hsv, self.blue_low, self.blue_high)
        blue_mask = cv2.dilate(blue_mask,kernel,iterations = 1)
        contours, hierarchy = cv2.findContours(blue_mask, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 20 and area < 300:
                x,y,w,h = cv2.boundingRect(cnt)
                if w > 5 and w < 60 and h > 5 and h < 60 and len(cnt) >= 5:
                    st = time.time()
                    ellipse = cv2.fitEllipse(cnt)
                    minor, major = ellipse[1]
                    if minor/major > 0.5:
                        # print("Image:", i)
                        # print("Ratio:",minor/major)
                        # print("Area:", area)
                        # print("Dimen:", w, h)
                        black_src = np.zeros((360,480,3), np.uint8)
                        # print(ellipse)
                        # ellipse_changed = ((ellipse[0][0] - x, ellipse[0][1] - y), ellipse[1], ellipse[2])
                        cv2.ellipse(black_src,ellipse,(255,255,255),-1)

                        # cv2.imshow("black src", black_src)
                        # crop = rgb[y:y+h, x:x+w]
                        mask = cv2.bitwise_and(black_src[y:y+h, x:x+w], rgb[y:y+h, x:x+w])
                        mask = cv2.resize(mask, (128,128))
                        pr.blue_mask = mask
                        saved_path = "/home/tuan/dira_data/test_sign_6_5/blue/img" + str(keyId1) + ".jpg"
                        cv2.imwrite(saved_path, mask)
                        keyId1 += 1

        red_mask = cv2.inRange(hsv, self.red_low, self.red_high)
        red_mask = cv2.dilate(red_mask,kernel,iterations = 1)
        contours, hierarchy = cv2.findContours(red_mask, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 20 and area < 300:
                x,y,w,h = cv2.boundingRect(cnt)
                if w > 5 and w < 60 and h > 5 and h < 60 and len(cnt) >= 5:
                    st = time.time()
                    ellipse = cv2.fitEllipse(cnt)
                    minor, major = ellipse[1]
                    if minor/major > 0.5:
                        # print("Image:", i)
                        # print("Ratio:",minor/major)
                        # print("Area:", area)
                        # print("Dimen:", w, h)
                        black_src = np.zeros((h,w,3), np.uint8)
                        ellipse_changed = ((ellipse[0][0] - x, ellipse[0][1] - y), ellipse[1], ellipse[2])
                        cv2.ellipse(black_src,ellipse_changed,(255,255,255),-1)
                        # crop = rgb[y:y+h, x:x+w]
                        mask = cv2.bitwise_and(black_src, rgb[y:y+h, x:x+w])
                        mask = cv2.resize(mask, (128,128))
                        pr.red_mask = mask
                        saved_path = "/home/tuan/dira_data/test_sign_6_5/red/img" + str(keyId2) + ".jpg"
                        cv2.imwrite(saved_path, mask)
                        keyId2 += 1

def main():
    img_proc = ImageProcessing()
    path = "/home/tuan/Downloads/data_full_3_12/rgb/img1710.jpeg"
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) 
    center_x, center_y, num_left, num_right = img_proc.detectLanes(gray, hsv)
    cv2.circle(img, (int(center_x),int(center_y)), 6, (0,0,255), -1)
    cv2.imwrite("/home/tuan/3.jpg", img)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()


    
    