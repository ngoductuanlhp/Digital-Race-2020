''' Python libs '''
import time
import numpy as np
from collections import Counter
import math

''' OpenCV libs '''
import cv2

''' Customized libs '''
import modules.param as par
# import param as par

class ImageProcessing:
    def __init__(self):

        self.white_low = np.array([0, 0, 180], dtype=np.uint8)
        self.white_high = np.array([150, 6, 255], dtype=np.uint8)

        self.verticalStructure1 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        self.verticalStructure2 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        self.horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
        self.horizontalStructure2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))

        # NOTE BEV param
        self.cutoff = 60
        pts1 = np.float32([[151, 120 - self.cutoff], [322, 120 - self.cutoff],
                        [0, 300 - self.cutoff], [480, 300 - self.cutoff]])
        pts2 = np.float32([[98, 120 - self.cutoff], [258, 120 - self.cutoff],
                        [98, 360 - self.cutoff], [258, 360 - self.cutoff]])
        self.M = cv2.getPerspectiveTransform(pts1, pts2)


    def colorFilterLine(self, img):
        h, w= img.shape[0], img.shape[1]
        white_mask = np.zeros(((h,w)), np.uint8)

        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        white_mask= cv2.inRange(hsv, self.white_low, self.white_high)

        ver_img = cv2.dilate(white_mask, self.verticalStructure1)
        hor_img = cv2.dilate(ver_img,self.horizontalStructure2)

        black_screen = np.zeros((h,w), np.uint8)
        contours, hie = cv2.findContours(hor_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= 900:
                cv2.drawContours(black_screen, [cnt], -1, 255, -1)
        return black_screen
    
    def checkMiddleVector(self, binary):
        mid_radius = 150 if not par.parking else 140
        vector_map = np.zeros((360,360), np.uint8)
        deg_90_point = (180, 360 - mid_radius)
        vector_map = cv2.line(vector_map,(180,360),deg_90_point,(255),10)

        vector_map = cv2.bitwise_and(binary, vector_map)
        if cv2.countNonZero(vector_map) <= 30:
            return False
        return True

    def detectSnow(self, binary):
        snow_roi = binary[110:340, 150:210]
        num = len([np.where(snow_roi == 255)][0][0])
        if num > 3500:
            return True
        return False

    def processLane(self, arr_x, arr_y, lane=1, mainLane=True):
        lane_angle = 0
        center_x = 180
        center_y = 150 if mainLane else 250

        nghieng_offset = 20 # NOTE Cam bi ngieng
        outside_offset = 30 if par.outside else 0
        if len(arr_x) > 0:
            center_y = arr_y[-1]
        if len(arr_x) > 1:
            ngang = -(arr_x[-1] - arr_x[0])
            doc = -(arr_y[-1] - arr_y[0])
            lane_angle = np.arctan2(ngang, doc) * 180/np.pi

            if lane_angle < -5:
                if par.timer_cursor in [0, 1, 3]:
                    angle_offset = -40
                elif par.timer_cursor in [2, 4]:
                    angle_offset = 20
                else:
                    angle_offset = 0
            else:
                angle_offset = lane_angle * 1.5
            angle_offset = angle_offset * np.sign(lane)

            if lane == 1:
                center_offset = (arr_x[-1] - 180) // 10
                if center_offset < 3:
                    center_x = int(arr_x[-1] - 50 - angle_offset - nghieng_offset - outside_offset)
                elif center_offset < 5:
                    center_x = int(arr_x[-1] - 40 - angle_offset - nghieng_offset - outside_offset)
                elif center_offset < 7:
                    center_x = int(arr_x[-1] - 35 - angle_offset - nghieng_offset - outside_offset)
                else:
                    center_x = int(arr_x[-1] - 20 - angle_offset - nghieng_offset - outside_offset)
            else:
                center_offset = (180 - arr_x[-1]) // 10
                if center_offset < 3:
                    center_x = int(arr_x[-1] + 50 - angle_offset - outside_offset)
                elif center_offset < 5:
                    center_x = int(arr_x[-1] + 40 - angle_offset - outside_offset)
                elif center_offset < 7:
                    center_x = int(arr_x[-1] + 35 - angle_offset - outside_offset)
                else:
                    center_x = int(arr_x[-1] + 20 - angle_offset - outside_offset)

        center = [max(0, center_x), center_y]
        return center, lane_angle

    def lane_tracking(self, binary, lane=1, interpolate=0, inside=True):
        par.outside = par.timer_cursor in [0, 1, 5]

        start_x = 180 if lane == 1 else 0
        start_x = 0 if interpolate == 1 else start_x
        end_x   = 360 if lane == 1 else 180

        sub_lane_x, main_lane_x = [], []
        sub_lane_y, main_lane_y = [], []
        num_sub_lane, num_main_lane = 0, 0
        
        null_lane = -1


        if par.parking:
            end_y = 280
        elif par.timer_cursor in [5, 6]:
            end_y = 240
        else:
            end_y = 200
       
        for idx in range(360, end_y, -10):
            if null_lane > 4:
                break
            
            top_window = binary[idx-10:idx-5, start_x:end_x]
            bot_window = binary[idx-5:idx, start_x:end_x]
            
            if (lane == -1 and not par.has_snow) or par.outside:
                top_window = top_window[:, ::-1]
                bot_window = bot_window[:, ::-1]
                top_window = top_window.transpose().flatten()
                bot_window = bot_window.transpose().flatten()
                
                top_lane = (end_x  - (np.where(top_window==255)[0][:5]//5)) 
                bot_lane = (end_x  - (np.where(bot_window==255)[0][:5]//5))
            else:
                top_window = top_window.transpose().flatten()
                bot_window = bot_window.transpose().flatten()
                top_lane = np.where(top_window==255)[0][:5]//5 + start_x 
                bot_lane = np.where(bot_window==255)[0][:5]//5 + start_x

            len_top, len_bot = len(top_lane), len(bot_lane)
            if len_top > 2 and len_bot > 2:
                null_lane = 0
                half_top_mean = np.mean(top_lane)
                half_bot_mean = np.mean(bot_lane)
                orient_offset = abs(half_bot_mean - half_top_mean)
                x_mean = (half_bot_mean + half_top_mean) / 2
                y_mean = idx - 5

                if idx > 310:
                    num_sub_lane += len_top + len_bot
                    sub_lane_x.append(x_mean)
                    sub_lane_y.append(y_mean)
                else:
                    num_main_lane += len_top + len_bot
                    main_lane_x.append(x_mean)
                    main_lane_y.append(y_mean)

                if lane == 1:
                    start_x = max(0, int(half_top_mean - 40 - orient_offset))
                    end_x   = min(360, int(half_top_mean + 70 + orient_offset))
                else:
                    start_x = max(0, int(half_top_mean - 70 - orient_offset))
                    end_x   = min(360, int(half_top_mean + 40 + orient_offset))
                
            else:
                if null_lane >= 0:
                    null_lane += 1
            
        main_center, main_lane_angle = self.processLane(main_lane_x, main_lane_y, lane=lane, mainLane=True)
        sub_center, sub_lane_angle = self.processLane(sub_lane_x, sub_lane_y, lane=lane, mainLane=False)
        return main_center, sub_center, num_main_lane, num_sub_lane, main_lane_angle, sub_lane_angle

    def find_intersection(self, line1, line2, max_x, max_y):
        rho1, theta1 = line1
        rho2, theta2 = line2
        A = np.array([
            [np.cos(theta1), np.sin(theta1)],
            [np.cos(theta2), np.sin(theta2)]
        ])
        b = np.array([[rho1], [rho2]])
        x0, y0 = np.linalg.solve(A, b)
        x0, y0 = int(np.round(x0)), int(np.round(y0))
        if y0 < 0 or y0 >= max_y:
            return None
        return [x0, y0 + 100]


    def segmented_intersections(self, v_lane, h_lane, size):
        intersections = []
        for vertical in v_lane:
            intersection = self.find_intersection(
                vertical, h_lane, size[1], size[0])
            if intersection:
                intersections.append(intersection)

        return intersections

    def timeToStop(self, pre_bev):
        threshhold = 150 if par.map_num == 2 else 180 
        offset = 2
        im = pre_bev[150:,:]
        midvec  = im[:,180 - offset: 180 + offset]

        if len(list(np.where(midvec == 255)[0])) == 0:
            print('stop null')
            if par.check_null_stop:
                return True
            else:
                return False
        par.check_null_stop = True
        point = max(list(np.where(midvec == 255)[0]))
        print('Stoppppppp', point)
        if point >= threshhold:
            return True
        # else:
        return False

    def detectStopLine(self, bev):
        im = cv2.cvtColor(bev, cv2.COLOR_RGB2GRAY)
        im = im[100:, :]
        _, threshold = cv2.threshold(im, 200, 255, cv2.THRESH_BINARY)

        '''Canny Edge Detection'''
        black = np.zeros(threshold.shape)
        canny = cv2.Canny(threshold, 80, 200)

        lines = cv2.HoughLines(canny, 1, np.pi/180, 20,
                            min_theta=-np.pi/4, max_theta=np.pi/4)

        if lines is None:
            return False, -1

        verticle_line = lines[0][0]
        v_line = None
        h_line = None
        lane = [(verticle_line[0], verticle_line[1])]


        '''Find Vertical Line'''
        for line in (lines[1:]):
            for rho, theta in line:
                if np.degrees(abs(theta - verticle_line[1])) <= 10 and abs(abs(verticle_line[0]) - abs(rho)) >= 100:
                    v_line = line
                    lane.append((rho, theta))
            if v_line is not None:
                break

        h_limit = []
        for line in lane:
            rho, theta = line
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            h_limit += [(rho)/a, (rho - b*360)/a]
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            cv2.line(black, (x1, y1), (x2, y2), 255, 2)

        if len(h_limit) == 4:
            left_limit = int(max(0, min(h_limit)))
            right_limit = int(min(360, max(h_limit)))
        else:
            left_limit = int(max(0, min(h_limit)))
            right_limit = 360
        canny_2 = canny.copy()[:, left_limit:right_limit]

        '''Find Horizontal Line'''
        lines = cv2.HoughLines(canny_2, 1, np.pi/180, 20, max_theta=verticle_line[1] - np.radians(70),
                            min_theta=verticle_line[1] - np.radians(110))
        if lines is None:
            return False, -1

        for rho, theta in lines[0]:
            h_line = (rho, theta)
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            cv2.line(black, (x1, y1), (x2, y2), 255, 2)
        intersections = self.segmented_intersections(lane, h_line, threshold.shape)

        if len(intersections) == 2:
            return True, ((intersections[0][0] + intersections[1][0])//2)
        return False, -1

    def projectBEV(self, img):
        dst = cv2.warpPerspective(img[self.cutoff:,:], self.M, (360, 360 - self.cutoff))
        dst = cv2.copyMakeBorder(dst, self.cutoff, 0, 0, 0, cv2.BORDER_CONSTANT)
        return dst

    def projectPointToBEV(self, point):
        point[1] = point[1] + self.cutoff
        homo_point = np.transpose(np.array([point[0],point[1],1])) 
        projected_point = np.dot(self.M, homo_point)
        projected_point[0:2] /= projected_point[2]
        return int(projected_point[0]), int(projected_point[1])
