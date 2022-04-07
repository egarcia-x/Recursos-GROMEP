from queue import Queue
from cv2 import aruco, cvtColor, COLOR_BGR2GRAY


def camera_thread(camera_pos_queue: Queue):
    marker_size = 66
    camera = globals()["camera"]
    aruco_dict = globals()["aruco_dict"]
    camera_matrix = globals()["camera_matrix"]
    camera_distortion = globals()["camera_distortion"]

    ret, frame = camera.read()

    gray = cvtColor(frame, COLOR_BGR2GRAY)

    corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, camera_matrix, camera_distortion)

    aruco.drawDetectedMarkers(frame, corners)

    if ids is not None and [47, 13, 36, 17, 42, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] in ids:
        rvec_list_all, tvec_list_all, _objPoints = aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, camera_distortion)

        camera_pos_queue.put(tvec_list_all)
