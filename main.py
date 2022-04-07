from threading import Thread
from queue import Queue

from numpy import load
from cv2 import aruco, VideoCapture

from camera import camera_thread
from calc_pos import calc_pos_thread
from operator import operator_thread


def main():
    with open('camera_cal.npy', 'rb') as f:
        globals()["camera_matrix"] = load(f)
        globals()["camera_distortion"] = load(f)

    globals()["aruco_dict"] = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)

    globals()["camera"] = VideoCapture(0)

    calc_op_queue: Queue = Queue()
    camera_calc_queue: Queue = Queue()

    thread_poll: map = map()
    thread_poll.append("camera", Thread(target=camera_thread, args=(camera_calc_queue), daemon=True))
    thread_poll.append("calc_pos", Thread(target=calc_pos_thread, args=(camera_calc_queue, calc_op_queue), daemon=True))
    thread_poll.append("operator", Thread(target=operator_thread, args=(calc_op_queue), daemon=True))

    thread_poll["camera"].start()
    thread_poll["calc_pos"].start()
    thread_poll["operator"].start()


if __name__ == "__main__":
    main()
