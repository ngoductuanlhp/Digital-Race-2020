import threading
import numpy as np

import pycuda.driver as cuda

import time
import rospy

from modules.yolo_detector import YoloDetector
import modules.param as par



class YoloThread(threading.Thread):
    """YoloThread

    This implements the child thread which continues to read images
    from cam (input) and to do TRT engine inferencing.  The child
    thread stores the input image and detection results into global
    variables and uses a condition varaiable to inform main thread.
    In other words, the TrtThread acts as the producer while the
    main thread is the consumer.
    """
    def __init__(self, eventStart, eventEnd):
        threading.Thread.__init__(self)
        
        self.eventStart = eventStart
        self.eventEnd = eventEnd

        self.cuda_ctx = None  # to be created when run
        self.trt_model = None   # to be created when run

        self.img = None

        self.boxes, self.scores, self.classid = None, None, None

    def run(self):
        """Run until 'running' flag is set to False by main thread.

        NOTE: CUDA context is created here, i.e. inside the thread
        which calls CUDA kernels.  In other words, creating CUDA
        context in __init__() doesn't work.
        """

        rospy.loginfo('[YoloThread] Loading the TRT model...')
        self.cuda_ctx = cuda.Device(0).make_context()  # GPU 0
        self.trt_model = YoloDetector()
        # Dummy input 
        dummy_inp = np.random.normal(size=(96,160,3)).astype(np.uint8)
        self.trt_model.infer(dummy_inp)

        rospy.loginfo('[YoloThread] Start running...')
        while not rospy.is_shutdown():
            self.eventStart.wait()
            self.eventStart.clear()
            if self.img is not None:
                self.boxes, self.scores, self.classid = self.trt_model.infer(self.img)
                self.img = None

            self.eventEnd.set()

        del self.trt_model
        self.cuda_ctx.pop()
        del self.cuda_ctx
        rospy.loginfo('[TrtThread] Stopped...')

    def stop(self):
        self.join()