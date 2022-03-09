#####
# Versión mejorada
#####

import cv2
import cv2.aruco as aruco
import numpy as np
import time
import math



#---------------------------------------------------------------------------------------------------------------
#----------- ROTATIONS https://www.learnopencv.com/rotation-matrix-to-euler-angles/
#---------------------------------------------------------------------------------------------------------------
# Checks if a matrix is a valid rotation matrix > y <

def isRotationMatrix(R):
	Rt = np.transpose(R)
	shouldBeIdentity = np.dot(Rt, R)
	I = np.identity(3, dtype=R.dtype)
	n = np.linalg.norm(I - shouldBeIdentity)
	return n < 1e-6
	
def rotationMatrixToEulerAngles(R):
	assert (isRotationMatrix(R))
	
	sy = math.sqrt(R[0, 0] * R[0, 0] + R[1,0] * R[1,0])
	singular = sy < 1e-6
	
	if not singular:
		x = math.atan2(R[2,1], R[2,2])
		y = math.atan2(-R[2,0], sy)
		z = math.atan2(R[1,0], R[0,0])
	else:
		x = math.atan2(-R[1,2], R[1,1])
		y = math.atan2(-R[2,0], sy)
		z = 0	
		
	return np.array([x,y,z])




####
prev_frame_time = time.time()

cal_image_count = 0
frame_count = 0
####


marker_size = 66

with open('assets/camera_cal.npy', 'rb') as f:
	camera_matrix = np.load(f)
	camera_distortion = np.load(f)
	
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)

cap = cv2.VideoCapture(0)


camera_width = 640
camera_height = 480
camera_frame_rate = 40

cap.set(2, camera_width)
cap.set(4, camera_height)
cap.set(5, camera_frame_rate)

id_to_find = 17


while True:
	
	ret, frame = cap.read()

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, camera_matrix, camera_distortion)
	

	if ids is not None and [32, 100, 69, 1] in ids:
	#if ids is not None and ids[0] == 32:   MAL OSTIA ESTÁS PUTO CIEGO JODERRRR
		
		for i in range(len(ids)):
			print('i')
			print(i)
			if ids[i] == [45]:
				pos = np.where(ids==[45])[0][0]
			elif ids[i] == [32]:
				pos = np.where(ids==[32])[0][0]
			#elif i == 69:
				#pos = np.where(ids==[69])[0][0]


				
		print("ids")
		print(ids)
		print(ids[0])
		
		#print("corners")
		#print(corners)
		
		#print("frame")
		#print(frame)
		#pos = np.where(ids==[32])[0][0]
		#print(pos)
		
		
		aruco.drawDetectedMarkers(frame, corners)
		
		rvec_list_all, tvec_list_all, _objPoints = aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, camera_distortion)
		#print('rvec_list_all')
		#print(rvec_list_all)
		rvec = rvec_list_all[pos][0]
		tvec = tvec_list_all[pos][0]
		#print('rvec')
		#print(rvec)
		rvec[0]=0
		rvec[1]=0  #Hemos hecho esta chapuza xq petaba cuando hacia cambios buscrus y/o cuando tenia x valores
		rvec[2]=0
		

		# Pienso que sería mejor que quitaramos esta funcion que realmente no nos aporta nada a parte de aesthetic
		aruco.drawAxis(frame, camera_matrix, camera_distortion, rvec, tvec, 100)
		
		
		new_frame_time = time.time()
		fps = 1/(new_frame_time - prev_frame_time)
		prev_frame_time = new_frame_time
		cv2.putText(frame, 'FPS ' + str(int(fps)), (10,40), cv2.FONT_HERSHEY_PLAIN, 3, (100,255,0), 2, cv2.LINE_AA)
		
		###
		rvec_flipped = rvec * -1
		tvec_flipped = tvec * -1
		rotation_matrix, jacobian = cv2.Rodrigues(rvec_flipped)
		realworld_tvec = np.dot(rotation_matrix, tvec_flipped)
		
		pitch, roll, yaw = rotationMatrixToEulerAngles(rotation_matrix)
			
		tvec_str = "x=%4.0f y=%4.0f direction=%4.0f"%(realworld_tvec[0], realworld_tvec[1], math.degrees(yaw))
		cv2.putText(frame, tvec_str, (20,400), cv2.FONT_HERSHEY_PLAIN, 2, (0,0, 255), 2, cv2.LINE_AA)

		###
		
		tvec_str = "x=%4.0f y=%4.0f z=%4.0f"%(tvec[0], tvec[1], tvec[2])
		cv2.putText(frame, tvec_str, (20,460), cv2.FONT_HERSHEY_PLAIN, 2, (0,0, 255), 2, cv2.LINE_AA)
		
	cv2.imshow('frame', frame)
		
	key = cv2.waitKey(1) & 0xFF
	if key == ord('q'): break
	
cap.release()
cv2.destroyAllWindows()
		
