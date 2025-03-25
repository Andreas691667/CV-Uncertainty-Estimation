from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2

from feature_definition import feature_specification

def convert_and_trim_bb(image, rect):
	# extract the starting and ending (x, y)-coordinates of the
	# bounding box
	startX = rect.left()
	startY = rect.top()
	endX = rect.right()
	endY = rect.bottom()
	# ensure the bounding box coordinates fall within the spatial
	# dimensions of the image
	startX = max(0, startX)
	startY = max(0, startY)
	endX = min(endX, image.shape[1])
	endY = min(endY, image.shape[0])
	# compute the width and height of the bounding box
	w = endX - startX
	h = endY - startY
	# return our bounding box coordinates
	return (startX, startY, w, h)

def ExtractFace(image, annotated_image, image_name, results_dir_name, image_summary):
	shape_predictor_file = "../featrues_specifications/person/shape_predictor_68_face_landmarks.dat"
	# initialize dlib's face detector (HOG-based) and then create
	# the facial landmark predictor
	# detector = dlib.get_frontal_face_detector()
	# predictor = dlib.shape_predictor(shape_predictor_file)
	# load dlib's CNN face detector
	print("[INFO] loading CNN face detector...")
	detector = dlib.cnn_face_detection_model_v1("../featrues_specifications/person/mmod_human_face_detector.dat")

	# load the input image, resize it, and convert it to grayscale
	# image = cv2.imread(image_path)
	# print(image)
	# image = imutils.resize(image, width=416, height=416)
	# image = imutils.resize(image, width=500)
	# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	print("[INFO[ performing face detection with dlib...")
	unsample_num = 2
	results = detector(rgb, unsample_num)

	# convert the resulting dlib rectangle objects to bounding boxes,
	# then ensure the bounding boxes are all within the bounds of the
	# input image
	boxes = [convert_and_trim_bb(image, r.rect) for r in results]
	
	# annotated_image = image.copy()
	# features_coordinates = []
	# loop over the bounding boxes
	for (x, y, w, h) in boxes:
		left = x
		top  = y
		right = x + w
		bottom = y + h

		# features_coordinates.append([left, top, right, bottom])
		image_summary.array_of_features.append(feature_specification(None, 0, left, top, right, bottom))
		
		# draw the bounding box on our image
		cv2.rectangle(annotated_image, (int(left), int(top)), (int(right), int(bottom)), (230, 0, 230), thickness=2)

	# annotated_image_path = results_dir_name+"04/"+image_name
	# cv2.imwrite(annotated_image_path, annotated_image)#cv2.flip(annotated_image, 1))
	




	# return features_coordinates

	# detect faces in the grayscale image
	# rects = detector(gray, 1)

	# print("rects = ",rects)

	# # loop over the face detections
	# annotated_image = image.copy()
	# features_coordinates = []
	# for (i, rect) in enumerate(rects):
	# 	# determine the facial landmarks for the face region, then
	# 	# convert the landmark (x, y)-coordinates to a NumPy array
	# 	shape = predictor(gray, rect)
	# 	shape = face_utils.shape_to_np(shape)
	# 	trigger = True

	# 	# loop over the face parts individually
	# 	print("face_utils.FACIAL_LANDMARKS_IDXS.items() = ",face_utils.FACIAL_LANDMARKS_IDXS.items())
	# 	for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
	# 		# clone the original image so we can draw on it, then
	# 		# display the name of the face part on the image
	# 		clone = image.copy()
	# 		cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
	# 			0.7, (0, 0, 255), 2)

	# 		# loop over the subset of facial landmarks, drawing the
	# 		# specific face part
	# 		for (x, y) in shape[i:j]:
	# 			cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)

	# 		# extract the ROI of the face region as a separate image
	# 		(x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))

	# 		# Work out the left, top, right, bottom parts here.
	# 		if trigger:
	# 			left = x
	# 			top  = y
	# 			right = x + w
	# 			bottom = y + h
	# 			trigger = False
	# 		left = np.min([x, left])
	# 		top = np.min([y, top])
	# 		right = np.max([x + w, right])
	# 		bottom = np.max([y + h, bottom])
			

	# 	features_coordinates.append([left, top, right, bottom])
	# 	cv2.rectangle(annotated_image, (int(left), int(top)), (int(right), int(bottom)), (0, 0, 230), thickness=2)
	# print("===============================================")
	# print("Original readings")
	# print("left = ", left)
	# print("top = ", top)
	# print("right = ", right)
	# print("bottom = ", bottom)
	# print("===============================================")
	
	# str_end = image_path[-4:]
	# annotated_image_path = image_path[:-4] + "_face_annotation" + str_end
	# cv2.imwrite('test.jpg', cv2.flip(annotated_image2, 1))
	# annotated_image_path = "04_face_annotation_cv2.png"
	# cv2.imwrite(annotated_image_path, annotated_image)#cv2.flip(annotated_image, 1))
	# return features_coordinates
			
# Test this fucntion --> Working
