from multiprocessing import Queue
import cv2
import math
import cv2.aruco as aruco
import numpy as np


def inversePerspective(rvec):
    R, _ = cv2.Rodrigues(rvec)
    R = np.matrix(R).T
    invRvec, _ = cv2.Rodrigues(R)
    return invRvec


def relativePosition(rvec1,rvec2):
    rvec1 = rvec1.reshape((3, 1))
    rvec2 = rvec2.reshape((3, 1))

    # Inverse the second marker, the right one in the image
    invRvec = inversePerspective(rvec2)
    
    composedRvec = cv2.composeRT(rvec1, invRvec)

    composedRvec = composedRvec.reshape((3, 1))
    
    return composedRvec



def calc_pos_thread(camera_calc_queue: Queue, calc_op_queue: Queue):
    while True:
        print("Iteration!")
        print(camera_calc_queue.get())
        calc_op_queue.put(camera_calc_queue.get())

