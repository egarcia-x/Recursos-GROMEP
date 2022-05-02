from asyncio.windows_events import NULL
from queue import Queue

from wheel_movement import WheelMovement

def operator_thread(calc_op_queue: Queue):
    # while(True):
    #    wheel = WheelMovement("192.168.0.181", 8001)

    #    result = calc_op_queue.get()

    #    if result != None:
    #        wheel.send_x_y_displacement(result[0], result[1])
    while True:
        calc_op_queue.get()
    