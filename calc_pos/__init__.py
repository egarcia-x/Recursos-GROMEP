from queue import Queue


def calc_pos_thread(camera_calc_queue: Queue, calc_op_queue: Queue):
    camera_calc_queue.get()

    calc_op_queue.put()

