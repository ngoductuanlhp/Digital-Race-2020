import cv2
import time
import threading

import tensorrt as trt

import numpy as np 
import pycuda.autoinit
import pycuda.driver as cuda


import modules.param as par
# import param as par


TRT_PATH = '/home/ml4u/important_dira/resnet_classifer_18_12.trt'
class SignDetector(object):
    def _load_plugins(self):
        trt.init_libnvinfer_plugins(self.trt_logger, '')

    def _load_engine(self, model_path):
        with open(model_path, 'rb') as f, trt.Runtime(self.trt_logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def _allocate_buffer(self):
        bindings = []
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            # print("szie:", size)
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            if self.engine.binding_is_input(binding):
                inp = {'host': host_mem, 'device': device_mem}
            else:
                out = {'host': host_mem, 'device': device_mem}
        return inp, out, bindings

    def __init__(self):

        self.blue_low    = np.array([100, 140, 40], dtype=np.uint8)
        self.blue_high   = np.array([120, 255, 200], dtype=np.uint8)

        self.red_low1     = np.array([170, 80, 40], dtype=np.uint8)
        self.red_high1    = np.array([180, 255, 200], dtype=np.uint8)

        self.red_low2     = np.array([0, 80, 40], dtype=np.uint8)
        self.red_high2    = np.array([10, 255, 200], dtype=np.uint8)

        self.blue_kernel = np.ones((11,11),np.uint8)
        self.red_kernel_erode = np.ones((3,3),np.uint8)
        self.red_kernel = np.ones((7,7),np.uint8)

        self.mean = np.stack((np.ones((64,64)) * 0.485*255.0,np.ones((64,64)) * 0.456*255.0,np.ones((64,64)) * 0.406*255.0))
        self.std = np.stack((np.ones((64,64)) * 0.229*255.0,np.ones((64,64)) * 0.224*255.0,np.ones((64,64)) * 0.225*255.0))

        self.sign_label = {0: "forward", 1: "left", 2: "right", 3: "stop", 4: "not_left", 5: "not_right", 6: "unknown_blue", 7:"unknown_red", 8: "noise"}
        self.threshold_red = 6
        self.threshold_blue = 5

        self.output_shape = (1,8)
        self.current_output = -1

        self.trt_logger = trt.Logger(trt.Logger.INFO)

        self.red_location = (-1,-1)
        

        try:
            self._load_plugins()
            self.engine = self._load_engine(TRT_PATH)
            self.context = self.engine.create_execution_context()
            self.stream = cuda.Stream()
            self.inp, self.out, self.binding = self._allocate_buffer()
        except Exception as e:
            raise RuntimeError('fail to allocate CUDA resources') from e

    def __del__(self):
        """Free CUDA memories and context."""
        del self.inp['device']
        del self.out['device']
        del self.stream

    def detect(self, img, depth):
        red_crop, blue_crop = None, None
        hsv         = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        blue_mask   = cv2.inRange(hsv, self.blue_low, self.blue_high)

        depth[depth == 0] = 255
        _, depth_mask = cv2.threshold(depth,70,255,cv2.THRESH_BINARY_INV)
        
        blue_mask   = cv2.bitwise_and(blue_mask, depth_mask)

        blue_mask   = cv2.dilate(blue_mask, self.blue_kernel, iterations = 1)
        contours, hierarchy = cv2.findContours(blue_mask, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # NOTE strict blue sign
            if area > 2500 and area < 10000:
                x,y,w,h = cv2.boundingRect(cnt)
                if w > 20 and w < 120 and h > 20 and h < 120:
                    ellipse = cv2.fitEllipse(cnt)
                    minor, major = ellipse[1]
                    if float(minor/major) >= 0.6:
                        self.red_location = (int(x+w/h), int(y+h/2))
                        black_src = np.zeros((h,w,3), np.uint8)
                        ellipse_changed = ((ellipse[0][0] - x, ellipse[0][1] - y), ellipse[1], ellipse[2])
                        cv2.ellipse(black_src,ellipse_changed,(255,255,255),-1)
                        blue_crop = cv2.bitwise_and(black_src, img[y:y+h, x:x+w])
                        blue_crop = cv2.resize(blue_crop, (64,64))
                        break

        red_mask1 = cv2.inRange(hsv, self.red_low1, self.red_high1)
        red_mask2 = cv2.inRange(hsv, self.red_low2, self.red_high2)
        red_mask = red_mask1 | red_mask2

        red_mask   = cv2.bitwise_and(red_mask, depth_mask)
        
        red_mask = cv2.erode(red_mask,self.red_kernel_erode)
        red_mask = cv2.dilate(red_mask,self.red_kernel,iterations = 1)
        contours, hierarchy = cv2.findContours(red_mask, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 600 and area < 10000 and len(cnt) >= 5:
                x,y,w,h = cv2.boundingRect(cnt)
                if w > 20 and w < 120 and h > 20 and h < 120:
                    ellipse = cv2.fitEllipse(cnt)
                    minor, major = ellipse[1]
                    if minor/major >= 0.6:
                        black_src = np.zeros((h,w,3), np.uint8)
                        ellipse_changed = ((ellipse[0][0] - x, ellipse[0][1] - y), ellipse[1], ellipse[2])
                        cv2.ellipse(black_src,ellipse_changed,(255,255,255),-1)
                        red_crop = cv2.bitwise_and(black_src, img[y:y+h, x:x+w])
                        red_crop = cv2.resize(red_crop, (64,64))
                        par.red_shape = (w, h)
                        break
        return blue_crop, red_crop
        

    def classify(self, inp, isBlue=False):
        self.inp['host'] = self.preprocess(img=inp)

        cuda.memcpy_htod_async(self.inp['device'],self.inp['host'], self.stream)
        self.context.execute_async(bindings=self.binding, stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.out['host'], self.out['device'], self.stream)
        self.stream.synchronize()

        final_output = self.postprocess(self.out['host'].reshape(self.output_shape), isBlue=isBlue)
        return final_output


    def preprocess(self, img):
        img = img.transpose(2,0,1)
        img = (img - self.mean)/self.std
        img = np.expand_dims(img, axis=0)
        return np.array(img, dtype=np.float32, order='C')

    def postprocess(self, out, isBlue):
        pred = np.argmax(out, 1)[0]
        if isBlue and out[0][pred] >= self.threshold_blue:
            return pred
        if not isBlue and out[0][pred] >= self.threshold_red:
            return pred
        return 8

    def handle(self, img, depth):
        blue_sign, red_sign = -1, -1
        blue_crop, red_crop = self.detect(img, depth)
        if blue_crop is not None:
            blue_sign = self.classify(blue_crop, isBlue=True)
        if red_crop is not None:
            red_sign = self.classify(red_crop)
            if red_sign == 3:
                par.red_location = self.red_location
        return blue_sign, red_sign


            
                
    
