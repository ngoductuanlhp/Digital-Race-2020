import threading
import numpy as np

import pycuda.driver as cuda

import time
import rospy

from modules.sign_detector import SignDetector
import modules.param as par



class SignThread(threading.Thread):
    """SignThread

    This implements the child thread which continues to read images
    from cam (input) and to do TRT engine inferencing.  The child
    thread stores the input image and detection results into global
    variables and uses a condition varaiable to inform main thread.
    In other words, the SignThread acts as the producer while the
    main thread is the consumer.
    """
    def __init__(self, eventStart, eventEnd):
        threading.Thread.__init__(self)
        
        self.eventStart = eventStart
        self.eventEnd = eventEnd

        self.cuda_ctx = None  # to be created when run
        self.trt_model = None   # to be created when run

        self.img = None
        self.depth = None

    def run(self):
        """Run until 'running' flag is set to False by main thread.

        NOTE: CUDA context is created here, i.e. inside the thread
        which calls CUDA kernels.  In other words, creating CUDA
        context in __init__() doesn't work.
        """

        print('[SignThread]: Loading the TRT model...')
        self.cuda_ctx = cuda.Device(0).make_context()  # GPU 0
        self.trt_model = SignDetector()
        # Dummy input 
        dummy_inp = np.random.normal(size=(64,64,3)).astype(np.uint8)
        self.trt_model.classify(dummy_inp)
        print('SignThread: start running...')
        while not rospy.is_shutdown():
            self.eventStart.wait()
            self.eventStart.clear()
            if self.img is not None:
                par.blue_sign, par.red_sign = self.trt_model.handle(self.img, self.depth)
                self.img = None

            self.eventEnd.set()

        del self.trt_model
        self.cuda_ctx.pop()
        del self.cuda_ctx
        print('SignThread: stopped...')

    def stop(self):
        self.join()