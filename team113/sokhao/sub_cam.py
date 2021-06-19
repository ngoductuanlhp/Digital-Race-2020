#!/usr/bin/env python
from __future__ import division
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import rospkg 

import numpy as np
import logging, os
import time

rospack = rospkg.RosPack()
path = rospack.get_path('team113')
bridge = CvBridge()
t = 0
keyId = 92

def rgb_callback(data):
    global t, keyId 
    cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
    cv2.imshow("Img", cv_image)
    cv2.waitKey(1)
    # cv2.imwrite('/home/tuan/DiRa_Data/sample_signs/img' + str(keyId) + '.jpg', cv_image)
    # keyId = keyId + 1
    # print("Time: %f" % (time.time() - t))
    # t = time.time()
    rospy.sleep(0.06)


def main():
    rospy.init_node('sub_camera', anonymous=True)
    rgb_image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, rgb_callback, queue_size = 1, buff_size=2**24)
    # rospy.Subscriber()
    t = time.time()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    cv2.destroyAllWindows()
if __name__ == '__main__':
    main()