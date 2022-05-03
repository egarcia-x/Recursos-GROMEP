from multiprocessing import Queue

from cv2 import aruco, VideoCapture, Rodrigues, circle, putText, FONT_HERSHEY_PLAIN, cvtColor, COLOR_BGR2GRAY, imshow, waitKey, destroyAllWindows, LINE_AA, composeRT
from numpy import matrix, dot, load
from math import pi, sqrt



def calc_pos_thread(camera_calc_queue: Queue, calc_op_queue: Queue):
    while True:
        print("Iteration!")
        print(camera_calc_queue.get())



        #calc_op_queue.put(camera_calc_queue.get())

