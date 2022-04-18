
import numpy as np
import cv2
import cv2.aruco as aruco
import glob
import argparse
import math

cap = cv2.VideoCapture(0)

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


firstMarkerID = 2
secondMarkerID = 7


with open('/home/pi/Desktop/Recursos-GROMEP/assets/camera_cal.npy', 'rb') as f:
	matrix_coefficients = np.load(f)
	distortion_coefficients = np.load(f)
	
	
def inversePerspective(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    R = np.matrix(R).T
    invTvec = np.dot(-R, np.matrix(tvec))
    invRvec, _ = cv2.Rodrigues(R)
    return invRvec, invTvec


def relativePosition(rvec1, tvec1, rvec2, tvec2):
    rvec1, tvec1 = rvec1.reshape((3, 1)), tvec1.reshape((3, 1))
    rvec2, tvec2 = rvec2.reshape((3, 1)), tvec2.reshape((3, 1))

    # Inverse the second marker, the right one in the image
    invRvec, invTvec = inversePerspective(rvec2, tvec2)

    orgRvec, orgTvec = inversePerspective(invRvec, invTvec)
    #print("rvec: ", rvec2, "tvec: ", tvec2, "\n and \n", orgRvec, orgTvec)

    info = cv2.composeRT(rvec1, tvec1, invRvec, invTvec)
    composedRvec, composedTvec = info[0], info[1]

    composedRvec = composedRvec.reshape((3, 1))
    composedTvec = composedTvec.reshape((3, 1))
    return composedRvec, composedTvec


def draw(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)
    # draw ground floor in green
    # img = cv2.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)
    # draw pillars in blue color
    for i,j in zip(range(4),range(4)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)
    # draw top layer in red color
    return img



pointCircle = (0, 0)
markerTvecList = []
markerRvecList = []

while True:
    
    ret, frame = cap.read()
    # operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Change grayscale
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)  # Use 5x5 dictionary to find markers
    parameters = aruco.DetectorParameters_create()  # Marker detection parameters

    # lists of ids and the corners beloning to each id
    corners, ids, rejected_img_points = aruco.detectMarkers(gray, aruco_dict, parameters=parameters, cameraMatrix=matrix_coefficients, distCoeff=distortion_coefficients)

    if np.all(ids is not None):  # If there are markers found by detector

        del markerTvecList[:]
        del markerRvecList[:]
        zipped = zip(ids, corners)
        ids, corners = zip(*(sorted(zipped)))
        axis = np.float32([[-0.01, -0.01, 0], [-0.01, 0.01, 0], [0.01, -0.01, 0], [0.01, 0.01, 0]]).reshape(-1, 3)
        for i in range(0, len(ids)):  # Iterate in markers
            # Estimate pose of each marker and return the values rvec and tvec---different from camera coefficients
            rvec, tvec, markerPoints = aruco.estimatePoseSingleMarkers(corners[i], 0.02, matrix_coefficients, distortion_coefficients)
            if ids[i] == firstMarkerID:
                firstRvec = rvec
                firstTvec = tvec
                isFirstMarkerCalibrated = True
                firstMarkerCorners = corners[i]

            elif ids[i] == secondMarkerID:
                secondRvec = rvec
                secondTvec = tvec
                isSecondMarkerCalibrated = True
                secondMarkerCorners = corners[i]

            # print(markerPoints)
            (rvec - tvec).any() # get rid of that nasty numpy value array error
            markerRvecList.append(rvec)
            markerTvecList.append(tvec)

            aruco.drawDetectedMarkers(frame, corners)  # Draw A square around the markers

        if len(ids) > 1:

            firstRvec, firstTvec = firstRvec.reshape((3, 1)), firstTvec.reshape((3, 1))
            #secondRvec, secondTvec = secondRvec.reshape((3, 1)), secondTvec.reshape((3, 1))  ES EN SERIO?!?!??!

            composedRvec, composedTvec = relativePosition(firstRvec, firstTvec, secondRvec, secondTvec)

        if len(ids) > 1 and composedRvec is not None and composedTvec is not None:

            info = cv2.composeRT(composedRvec, composedTvec, secondRvec.T, secondTvec.T)
            TcomposedRvec, TcomposedTvec = info[0], info[1]
            moduleRvec = (math.sqrt(composedRvec[0][0]**2 + composedRvec[1][0]**2 + composedRvec[2][0]**2))*(180/math.pi)
            #print(composedTvec)
            moduleTvec = (math.sqrt(composedTvec[0][0]**2 + composedTvec[1][0]**2 + composedTvec[2][0]**2))
            #print('module_rvec: ', moduleRvec)
            print('module_tvec:  ',  moduleTvec)
            objectPositions = np.array([(0, 0, 0)], dtype=np.float)  # 3D point for projection
            
            imgpts, jac = cv2.projectPoints(axis, TcomposedRvec, TcomposedTvec, matrix_coefficients, distortion_coefficients)

            # frame = draw(frame, corners[0], imgpts)
            aruco.drawAxis(frame, matrix_coefficients, distortion_coefficients, TcomposedRvec, TcomposedTvec, 0.01)  # Draw Axis
            relativePoint = (int(imgpts[0][0][0]), int(imgpts[0][0][1]))
            cv2.circle(frame, relativePoint, 2, (255, 255, 0))


            cv2.putText(frame, 'distance: ' + str(int(moduleTvec*100)), (40, 400), cv2.FONT_HERSHEY_PLAIN, 2.5, (100,255,0), 2, cv2.LINE_AA)
            cv2.putText(frame, 'angle: ' + str(int(moduleRvec)), (40, 450), cv2.FONT_HERSHEY_PLAIN, 2.5, (100,255,0), 2, cv2.LINE_AA)


    # Display the resulting frame
    cv2.imshow('frame', frame)
    # Wait 3 milisecoonds for an interaction. Check the key and do the corresponding job.
    key = cv2.waitKey(3) & 0xFF
    if key == ord('q'):  # Quit
        break

    # When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
