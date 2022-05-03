from multiprocessing import Queue

from cv2 import aruco, VideoCapture, Rodrigues, circle, putText, FONT_HERSHEY_PLAIN, cvtColor, COLOR_BGR2GRAY, imshow, waitKey, destroyAllWindows, LINE_AA, composeRT
from numpy import matrix, dot, load, linalg
from math import pi, sqrt

def inversePerspective(rvec, tvec):
    R, _ = Rodrigues(rvec)
    R = matrix(R).T
    invTvec = dot(-R, matrix(tvec))
    invRvec, _ = Rodrigues(R)
    return invRvec, invTvec


def relativePosition(rvec1, tvec1, rvec2, tvec2):
    rvec1, tvec1 = rvec1.reshape((3, 1)), tvec1.reshape((3, 1))
    rvec2, tvec2 = rvec2.reshape((3, 1)), tvec2.reshape((3, 1))

    invRvec, invTvec = inversePerspective(rvec2, tvec2)

    info = composeRT(rvec1, tvec1, invRvec, invTvec)
    composedRvec, composedTvec = info[0], info[1]

    composedRvec = composedRvec.reshape((3, 1))
    composedTvec = composedTvec.reshape((3, 1))
    return composedRvec, composedTvec





def camera_thread(camera_calc_queue: Queue):
    from multiprocessing import Queue

    
    with open('/home/pi/Desktop/Recursos-GROMEP/assets/camera_cal.npy', 'rb') as f:
        camera_matrix = load(f)
        camera_distortion = load(f)
        
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)

    cap = VideoCapture(0)

    purpleDict = {}
    yellowDict = {}
    markersDict = {}

    yellow_id = False
    marker_detected = False
    purple_id = False

    distance_robot = 1000

    ROBOT = 1 # 0 es morado y 1 es amarillo

    while True:
        
        ret, frame = cap.read()

        gray = cvtColor(frame, COLOR_BGR2GRAY)

        corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, camera_matrix, camera_distortion)

        aruco.drawDetectedMarkers(frame, corners)

        if ids is not None:
        
            for i, j, l, L in zip(ids, corners, range(0, len(ids)), range(0, len(ids)-1)):
                rvecs, tvecs, markerPoints = aruco.estimatePoseSingleMarkers(corners[l], 0.02, camera_matrix, camera_distortion)
                (rvecs-tvecs).any()

                if ids[l] in range(1, 5):
                    purple_id = True
                    cx_purple, cy_purple = int(int(j[0][0][0] + j[0][1][0] + j[0][2][0] + j[0][3][0])/4), int(int(j[0][0][1] + j[0][1][1] + j[0][2][1] + j[0][3][1])/4)
                    circle(frame, (cx_purple, cy_purple), 3, (0,255,0), 2)
                    purpleDict["purple"] = [{"ID": i, "center": {"x": cx_purple, "y": cy_purple}}]
                    putText(frame, str(purpleDict), (10, 450), FONT_HERSHEY_PLAIN, 1, (100,255,0), 2, LINE_AA)
                    rvec_p, tvec_p = rvecs, tvecs

                    rvec_p, tvec_p = rvec_p.reshape((3,1)), tvec_p.reshape((3,1))
                    

                elif ids[l] in range(6, 10):
                    yellow_id = True
                    cx_yellow, cy_yellow = int(int(j[0][0][0] + j[0][1][0] + j[0][2][0] + j[0][3][0])/4), int(int(j[0][0][1] + j[0][1][1] + j[0][2][1] + j[0][3][1])/4)
                    circle(frame, (cx_yellow, cy_yellow), 3, (0,255,0), 2)
                    yellowDict["yellow"] = [{"ID": i, "center": {"x": cx_yellow, "y": cy_yellow}}]
                    putText(frame, str(yellowDict), (10, 450), FONT_HERSHEY_PLAIN, 1, (100,255,0), 2, LINE_AA)
                    rvec_y, tvec_y = rvecs, tvecs

                    rvec_y, tvec_y = rvec_y.reshape((3,1)), tvec_y.reshape((3,1))

                elif 11 < i < 50:
                    marker_detected = True

                if purple_id and marker_detected and ROBOT == 0:
                    if 11 < ids[l] < 50:

                        rvec, tvec = rvecs, tvecs
                        rvec, tvec = rvec.reshape((3,1)), tvec.reshape((3,1))
                        composedRvec, composedTvec = relativePosition(rvec_p, tvec_p, rvec, tvec)
                        moduleRvec = (sqrt(composedRvec[0][0]**2 + composedRvec[1][0]**2 + composedRvec[2][0]**2))*(180/pi)
                        cx, cy = int(int(j[0][0][0] + j[0][1][0] + j[0][2][0] + j[0][3][0])/4), int(int(j[0][0][1] + j[0][1][1] + j[0][2][1] + j[0][3][1])/4)
                        circle(frame, (cx, cy), 2, (0,255,0), 1)
                        distance_p = cx - cx_purple, cy - cy_purple
                        markersDict["aruco_robot{0}".format(l)] = [{"ID": i, "distance": {"x":distance_p[0], "y":distance_p[1]}, "heading": moduleRvec}]

                    if yellow_id:
                        distance_robot = sqrt(((cx_purple - cx_yellow)**2) + ((cy_purple - cy_yellow)**2))
                        markersDict['robot_distance'] = distance_robot                    

                if yellow_id and marker_detected and ROBOT == 1:
                    if 11 < ids[l] < 50:

                        rvec, tvec = rvecs, tvecs
                        rvec, tvec = rvec.reshape((3,1)), tvec.reshape((3,1))
                        composedRvec, composedTvec = relativePosition(rvec_y, tvec_y, rvec, tvec)
                        moduleRvec = (sqrt(composedRvec[0][0]**2 + composedRvec[1][0]**2 + composedRvec[2][0]**2))*(180/pi)

                        cx, cy = int(int(j[0][0][0] + j[0][1][0] + j[0][2][0] + j[0][3][0])/4), int(int(j[0][0][1] + j[0][1][1] + j[0][2][1] + j[0][3][1])/4)
                        circle(frame, (cx, cy), 2, (0,255,0), 1)
                        distance_y = cx - cx_yellow, cy - cy_yellow
                        markersDict["aruco_robot{0}".format(l)] = [{"ID": i, "center": {"x":distance_y[0], "y":distance_y[1]}, "heading": moduleRvec}]
                       
                    if purple_id:
                        distance_robot = sqrt(((cx_purple - cx_yellow)**2) + ((cy_purple - cy_yellow)**2))
                        markersDict['robot_distance'] = distance_robot               

                camera_calc_queue.put(markersDict)

        imshow('frame', frame)
            
        key = waitKey(1) & 0xFF
        if key == ord('q'): break
        
    cap.release()
    destroyAllWindows()
