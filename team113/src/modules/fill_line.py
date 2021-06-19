import cv2
import numpy as np
import modules.param as par

def fill_lane_bev(img, prev_angle=0, angle_offset=np.pi/24, rho_offset=None, part=0):
    test = img.copy()
    start_roi = 60
    stop_roi = 360
    width = 360
    if part == 1:
        img[:, :144,:] = 0
    elif part == -1:
        img[:, 216:,:] = 0

    if prev_angle > 0:
        # prev_angle = np.pi - prev_angle
        min_theta = prev_angle - angle_offset
        max_theta = prev_angle + angle_offset
    else:
        # prev_angle = -prev_angle
        min_theta = prev_angle - angle_offset
        max_theta = prev_angle + angle_offset

    im = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    im = im[start_roi:stop_roi, :]
    _, threshold = cv2.threshold(im, 180, 255, cv2.THRESH_BINARY)
    '''Canny Edge Detection'''
    black = np.zeros(threshold.shape)
    canny = cv2.Canny(threshold, 80, 200)
    # cv2.imshow('canny', canny)
    lines = cv2.HoughLines(canny, 1, np.pi/180, 15,
                           min_theta=min_theta, max_theta=max_theta)

    if lines is None:
        black = black[20:,:]
        final = np.vstack([np.zeros((80, width)), black])
        return final, 0

    rho_offset = abs(par.interpolate_offset*np.cos(prev_angle))
    lane = None
    current_rho = 0
    current_theta = 0
    _, standard_theta = lines[0][0]

    for line in lines:
        for rho, theta in line:
            # print(rho, np.degrees(theta))
            if rho_offset is not None:
                if par.bam_lane == 1:
                    if abs(rho)-rho_offset > 10 or abs(rho)-rho_offset < -40:
                    # if abs(abs(rho)-rho_offset) > 20 :
                        continue
                else:
                    if abs(rho)-rho_offset < -10 or abs(rho)-rho_offset > 40:
                    # if abs(abs(rho)-rho_offset) > 20 :
                        continue
            current_rho = abs(rho)
            current_theta = theta
            # print(rho, theta)
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            cv2.line(black, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.line(test, (x1, y1 + 100), (x2, y2 + 100), (255, 0, 0), 2)
            lane = line
            # cv2.imshow('test', test)
        if lane is not None:
            break

    if lane is None:
        theta, rho = standard_theta, rho_offset
        current_rho = rho_offset
        current_theta = theta
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(black, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.line(test, (x1, y1 + 100), (x2, y2 + 100), (255, 0, 0), 2)

        x0 = -a*rho
        y0 = -b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(black, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.line(test, (x1, y1 + 100), (x2, y2 + 100), (255, 0, 0), 2)

    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
    black = cv2.dilate(black, horizontalStructure)
    black = black[20:,:]
    final = np.vstack([np.zeros((80, width)), black])
    # cv2.imshow('res', test)
    par.interpolate_offset = current_rho
    par.prev_lane_angle = current_theta

    return final, current_rho

def fill_lane(img, prev_angle = 3*np.pi/16, angle_offset = np.pi/16):
    test = img.copy()
    start_roi = 60
    stop_roi = 315
    width = 480
    img[:,:160] = 0

    if prev_angle > 0:
        prev_angle = np.pi - prev_angle
        min_theta = prev_angle - angle_offset
        max_theta = max(np.pi, prev_angle + angle_offset)
    else:
        prev_angle = -prev_angle
        min_theta = min(0, prev_angle - angle_offset)
        max_theta = prev_angle + angle_offset

    im = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    im = im[start_roi:stop_roi, :]
    _, threshold = cv2.threshold(im, 180, 255, cv2.THRESH_BINARY)
    '''Canny Edge Detection'''
    black = np.zeros(threshold.shape)
    canny = cv2.Canny(threshold, 80, 200)
    # cv2.imshow('canny', canny)
    lines = cv2.HoughLines(canny, 1, np.pi/180, 10,
                           min_theta=min_theta, max_theta=max_theta)
    if lines is None:
        return black

    for rho, theta in lines[0]:
        print(np.degrees(theta))
        # if abs(rho) > rho_threshold:
        #     continue
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(black, (x1, y1), (x2, y2), 255, 2)
        cv2.line(test, (x1, y1 + 60), (x2, y2 + 60), (255, 0, 0), 2)
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
    black = cv2.dilate(black, horizontalStructure)
    black = black[20:, :]
    final = np.vstack([np.zeros((80, width)), black,  np.zeros((45, width))])
    # cv2.imshow('test', test)
    return final

