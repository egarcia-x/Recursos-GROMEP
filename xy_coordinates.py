import cv2
from cv2 import cornerHarris
import cv2.aruco as aruco
import numpy as np
import math
#from helpers import distance, order_points
def isRotationMatrix(R):
    Rt =np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype= R.dtype)
    n = np.linalg.norm(I- shouldBeIdentity)
    return n < 1e-6


def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6
    
    if not singular:
        x = math.atan2(R[2,1], R[2,2])
        y = math.atan2(-R[2,0],sy)
        z = math.atan2(R[1,0], R[0,0])
    else:
        x = math.atan2(-R[1,2],R[1,1])
        y = math.atan2(-R[2,0],sy)
        z = 0 

    return np.array([x, y, z])


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


    #print("rvec: ", rvec2, "tvec: ", tvec2, "\n and \n", orgRvec, orgTvec)
    print(rvec1)
    print(invRvec)
    composedRvec = cv2.composeRT(rvec1, invRvec)

    composedRvec = composedRvec.reshape((3, 1))
    
    return composedRvec



with open('/home/pi/Desktop/Recursos-GROMEP/assets/camera_cal.npy', 'rb') as f:
	camera_matrix = np.load(f)
	camera_distortion = np.load(f)
	
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)

cap = cv2.VideoCapture(0)

markerDict = {}
robot_heading = 0
robotID = 7
targetID = 32

robotDetected = False
targetDetected = False
sex = False
robotDict = {}

while True:
	
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    xarx = 0
    corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, camera_matrix, camera_distortion)
    markersDict = {}
    #values2 = {"center": {"x": cx, "y": cy}, "heading": head, "markerCorners": [{"x": corners[0], "y": corners[0]}, {"x": corners[1], "y": corners[1]}, {"x": corners[2], "y": corners[2]}, {"x": corners[3], "y": corners[3]}], "size": size}
    l = 0
    if ids is not None:
        for i,j,l in zip(ids, corners, range(len(ids))):
            
            if i in range(5,10):
                rvec_r, tvec_r, markerPoints = aruco.estimatePoseSingleMarkers(j, 0.02, camera_matrix, camera_distortion)

                robotDetected = True
                cx = int(int(j[0][0][0] + j[0][1][0] + j[0][2][0] + j[0][3][0])/4)
                cy = int(int(j[0][0][1] + j[0][1][1] + j[0][2][1] + j[0][3][1])/4)
                markersDict["aruco_robot"] = [{"ID": i, "center": {"x": cx, "y": cy}, "markerCorners": [{"x1": j[0][0][0], "y1": j[0][0][1]}, {"x2": j[0][1][0], "y2": j[0][1][1]}, {"x3": j[0][2][0], "y3": j[0][2][1]}, {"x4":j[0][3][0], "y4":j[0][3][1]}]}]
                cv2.circle(frame, (cx, cy), 2, (0,255,0), 1)
                robot_center_position = np.array([cx,cy])
                rvec_r_flipped = rvec_r * -1
                rotation_matrix, jacobian = cv2.Rodrigues(rvec_r_flipped)
                pitch, roll, yaw = rotationMatrixToEulerAngles(rotation_matrix)
                robot_heading= math.degrees(yaw)
            elif i > 11:
                targetDetected = True
                rvec, tvec, markerPoints = aruco.estimatePoseSingleMarkers(j, 0.02, camera_matrix, camera_distortion)

                cx = int(int(j[0][0][0] + j[0][1][0] + j[0][2][0] + j[0][3][0])/4)
                cy = int(int(j[0][0][1] + j[0][1][1] + j[0][2][1] + j[0][3][1])/4)
                current_target_position = np.array([cx, cy])
                #print(cx, cy)
                #markersDict["aruco{0}".format(l)] = [{"ID": i, "center": {"x": cx, "y": cy}, "markerCorners": [{"x1": j[0][0][0], "y1": j[0][0][1]}, {"x2": j[0][1][0], "y2": j[0][1][1]}, {"x3": j[0][2][0], "y3": j[0][2][1]}, {"x4":j[0][3][0], "y4":j[0][3][1]}]}]
                markersDict["aruco"] = [{"ID": i, "center": {"x": cx, "y": cy}, "markerCorners": [{"x1": j[0][0][0], "y1": j[0][0][1]}, {"x2": j[0][1][0], "y2": j[0][1][1]}, {"x3": j[0][2][0], "y3": j[0][2][1]}, {"x4":j[0][3][0], "y4":j[0][3][1]}]}]

                #print(markersDict)
                
                cv2.circle(frame, (cx, cy), 2, (0,255,0), 1)

                aruco_markers = markersDict["aruco"]
                for m in aruco_markers:
                    x = int(m['center']['x'])
                    y = int(m['center']['y'])
            
            if robotDetected and targetDetected:
                print(rvec_r[0])
                
                #composedRvec = relativePosition(rvec_r[0][0], rvec[0][xarx])
                robot_distance_to_target = np.linalg.norm(robot_center_position - current_target_position)
                #moduleRvec = (math.sqrt(composedRvec[0][0]**2 + composedRvec[1][0]**2 + composedRvec[2][0]**2))*(180/math.pi)

                xarx=xarx+1
                    #robot_distance_to_target = np.linalg.norm(robot_center_position - current_target_position)
                   
                robotDict["aruco{0}".format(l)] = [{"ID": i, "module": robot_distance_to_target}]
                print(robotDict)
                #robot_distance_to_target_vec= np.linalg.norm(tvec- tvec_r)
                #print('robot_distance_to_target_vec \n')
                #print(robot_distance_to_target_vec)
                #print('robot_distance_to_target'+ '\n')
                #print(robot_distance_to_target)
                #print(robot_center_position - current_target_position)

    cv2.imshow('', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break

cap.release()
cv2.destroyAllWindows()
