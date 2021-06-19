from primesense import openni2
from primesense import _openni2 as c_api
import numpy as np
import cv2
import rospy
import threading

class Camera:
    def __init__(self):
        # try:
        openni2.initialize('/home/ml4u/catkin_ws/src/dira_package/team113/src/modules')

        self.dev = openni2.Device.open_any()

        self.rgb_stream = self.dev.create_color_stream()
        self.rgb_stream.set_video_mode(c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_RGB888, resolutionX=640, resolutionY=480, fps=30))
        
        self.depth_stream = self.dev.create_depth_stream()
        self.depth_stream.set_video_mode(c_api.OniVideoMode(pixelFormat = c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_100_UM, resolutionX=640, resolutionY=480, fps = 30))
            
        self.dev.set_depth_color_sync_enabled(True)
        self.dev.set_image_registration_mode(openni2.IMAGE_REGISTRATION_DEPTH_TO_COLOR)

        self.rgb_stream.start()
        self.depth_stream.start()

        self.rgb_image = np.zeros((360, 480, 3), np.uint8)
        self.depth_image = np.zeros((360, 480), np.uint8)
        self.sync_rgb_image = False
        # except:
        #     print("init Camera ERROR")

    def getRGB(self, lock_rgb):
        rospy.loginfo("[RGBThread] Start")
        while not rospy.is_shutdown():
            try:
                rgb = np.fromstring(self.rgb_stream.read_frame().get_buffer_as_uint8(),dtype=np.uint8).reshape(480,640,3)
                rgb = cv2.resize(rgb, dsize=(480,360))
                lock_rgb.acquire()
                self.rgb_image = cv2.flip(rgb, 1)
                lock_rgb.release()
                self.sync_rgb_image = True
            except:
                rospy.loginfo("[RGBThread] ERROR")
                self.rgb_image = np.zeros((360, 480, 3), np.uint8)
            

    def getDepth(self, lock_depth):
        while not rospy.is_shutdown():
            try:
                dmap = np.fromstring(self.depth_stream.read_frame().get_buffer_as_uint16(),dtype=np.uint16).reshape(480,640)  # Works & It's FAST
                dmap = cv2.resize(dmap, (480, 360))
                dmap = (dmap * (255/64892)).astype(np.uint8)
                lock_depth.acquire()
                self.depth_image = cv2.flip(dmap, 1)
                lock_depth.release()
            except:
                print("GET DEPTH ERROR")
                self.depth_image = np.zeros((360, 480), np.uint8)

    def shutdown(self):
        self.rgb_stream.stop()
        self.depth_stream.stop()
        openni2.unload()
  



    



