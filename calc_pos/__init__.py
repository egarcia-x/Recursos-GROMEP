from multiprocessing import Queue


def calc_pos_thread(camera_calc_queue: Queue, calc_op_queue: Queue):
    while True:
        print("Iteration!")
        print(camera_calc_queue.get())
