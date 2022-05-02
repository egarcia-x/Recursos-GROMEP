from multiprocessing import Process, Queue
from time import sleep
from camera import camera_thread
from calc_pos import calc_pos_thread
#from operator import operator_thread


def main():
    calc_op_queue: Queue = Queue()
    camera_calc_queue: Queue = Queue()

    thread_poll: dict = dict()
    thread_poll["camera"] = Process(name = "camera", target=camera_thread, args=(((camera_calc_queue), )), daemon=True)
    thread_poll["calc_pos"] = Process(name = "calc_pos", target=calc_pos_thread, args=(((camera_calc_queue), (calc_op_queue), )), daemon=True)
    #thread_poll["operator"] = Process(name = "operator", target=operator_thread, args=(((calc_op_queue), )), daemon=True)

    thread_poll["camera"].start()
    #thread_poll["camera"].join()

    thread_poll["calc_pos"].start()
    #thread_poll["calc_pos"].join()

    #thread_poll["operator"].start()
    #thread_poll["operator"].join()

    while(True):
        sleep(1)

if __name__ == "__main__":
    main()
