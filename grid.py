import cv2
import cv2.aruco as aruco
import numpy as np
import time
import math

with open('/home/pi/Desktop/Recursos-GROMEP/assets/camera_cal.npy', 'rb') as f:
	camera_matrix = np.load(f)
	camera_distortion = np.load(f)
	
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)

cap = cv2.VideoCapture(0)

marker_size = 5

while True:
	
	ret, frame = cap.read()

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, camera_matrix, camera_distortion)
	corn = corners
	aruco.drawDetectedMarkers(frame, corners)
	
	if len(corners) > 0:
		ids = ids.flatten()
		
		for (markerCorner, markerID, i) in zip(corners, ids, range(len(ids))):
			
			rvecs, tvecs, _objPoints = aruco.estimatePoseSingleMarkers(corn, marker_size, camera_matrix, camera_distortion)
			
			corners = markerCorner.reshape((4,2))
			(topLeft, topRight, bottomRight, bottomLeft) = corners

			topRight = (int(topRight[0]), int(topRight[1]))
			bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
			bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
			topLeft = (int(topLeft[0]), int(topLeft[1]))
			
			# draw the bounding box of the ArUCo detection
			cv2.line(frame, topLeft, topRight, (0, 255, 0), 2)
			cv2.line(frame, topRight, bottomRight, (0, 255, 0), 2)
			cv2.line(frame, bottomRight, bottomLeft, (0, 255, 0), 2)
			cv2.line(frame, bottomLeft, topLeft, (0, 255, 0), 2)
			
			
			# compute and draw the center (x, y)-coordinates of the ArUco
			# marker
			cX = int((topLeft[0] + bottomRight[0]) / 2.0)
			cY = int((topLeft[1] + bottomRight[1]) / 2.0)
			cv2.circle(frame, (cX, cY), 4, (0, 0, 255), -1)
			
			# draw the ArUco marker ID on the image
			#print(tvec_list_all)
			cv2.putText(frame, str(markerID),
				(topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
				0.5, (0, 255, 0), 2)

			cv2.putText(frame, str(tvecs[i]), (bottomLeft[0], bottomLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
			for i in range(len(ids)):
				cv2.aruco.drawAxis(frame, camera_matrix, camera_distortion, rvecs[i], tvecs[i], 3)
		
			
			with open ('/home/pi/Desktop/Recursos-GROMEP/data.csv', 'a') as datafile:
				datafile.write(str(tvecs) + '\n')
				
				
			#print(corners)		
			#print("[INFO] ArUco marker ID: {}".format(markerID))
		
			
#https://stackoverflow.com/questions/1060090/changing-variable-names-with-python-for-loops
# ><
	cv2.imshow('frame', frame)
		
	key = cv2.waitKey(1) & 0xFF
	if key == ord('q'): break
	
cap.release()
cv2.destroyAllWindows()
