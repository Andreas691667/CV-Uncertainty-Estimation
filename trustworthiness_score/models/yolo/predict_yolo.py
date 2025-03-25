
# load yolov3 model and perform object detection
# based on https://github.com/experiencor/keras-yolo3
import numpy as np
import math
from numpy import expand_dims
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

import cv2
import sys
# sys.path.append("..") # Adds higher directory to python modules path.

from yolo.get_yolo_prediction import *


# ============================================================================================================================================
# === Loading trained model and getting prediction
# ============================================================================================================================================

# define our new photo
# photo_filename = 'person_001.jpg'#'person_069.jpg' # 'zebra.jpg' 
def YoloPredict(image, input_w, input_h,min_class_threshold):
	# load yolov3 model
	model = load_model('../models/yolo/model.h5')
	# define the expected input shape for the model
	# input_w, input_h = 416, 416

	# load and prepare image
	# image, image_w, image_h = load_image_pixels(image_path, (input_w, input_h))
	

	# make prediction
	yhat = model.predict(image)

	# # summarize the shape of the list of arrays
	# print([a.shape for a in yhat])
	# define the anchors
	anchors = [[116,90, 156,198, 373,326], [30,61, 62,45, 59,119], [10,13, 16,30, 33,23]]

	# define the probability threshold for detected objects
	class_threshold = min_class_threshold
	boxes = list()
	for i in range(len(yhat)):
		# decode the output of the network
		boxes += decode_netout(yhat[i][0], anchors[i], class_threshold, input_h, input_w)

	# correct the sizes of the bounding boxes for the shape of the image
	# correct_yolo_boxes(boxes, image_h, image_w, input_h, input_w)
	correct_yolo_boxes(boxes, input_h, input_w, input_h, input_w)

	# suppress non-maximal boxes
	do_nms(boxes, 0.5)

	# define the labels
	labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
		"boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
		"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
		"backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
		"sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
		"tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
		"apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
		"chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
		"remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
		"book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

	# get the details of the detected objects
	v_boxes, v_labels, v_scores, box_classes_scores = get_boxes(boxes, labels, class_threshold)
	

	# str_end = image_path[-4:]
	# annotated_image_path = image_path[:-4] + "_prediction_annotation" + str_end
	# prediction_coordinates = draw_boxes(image, image_path, v_boxes, v_labels, v_scores, annotated_image_path)

	return v_boxes, v_labels, v_scores, box_classes_scores


def YoloPredict_fromPath(image_path, input_w, input_h):
	# load yolov3 model
	model = load_model('../models/yolo/model.h5')
	# define the expected input shape for the model
	# input_w, input_h = 416, 416

	# load and prepare image
	image, image_w, image_h = load_image_pixels(image_path, (input_w, input_h))
	

	# make prediction
	yhat = model.predict(image)

	# # summarize the shape of the list of arrays
	# print([a.shape for a in yhat])
	# define the anchors
	anchors = [[116,90, 156,198, 373,326], [30,61, 62,45, 59,119], [10,13, 16,30, 33,23]]

	# define the probability threshold for detected objects
	class_threshold = 0.6
	boxes = list()
	for i in range(len(yhat)):
		# decode the output of the network
		boxes += decode_netout(yhat[i][0], anchors[i], class_threshold, input_h, input_w)

	# correct the sizes of the bounding boxes for the shape of the image
	# correct_yolo_boxes(boxes, image_h, image_w, input_h, input_w)
	correct_yolo_boxes(boxes, input_h, input_w, input_h, input_w)

	# suppress non-maximal boxes
	do_nms(boxes, 0.5)

	# define the labels
	labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
		"boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
		"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
		"backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
		"sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
		"tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
		"apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
		"chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
		"remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
		"book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

	# get the details of the detected objects
	v_boxes, v_labels, v_scores, box_classes_scores = get_boxes(boxes, labels, class_threshold)
	

	str_end = image_path[-4:]
	annotated_image_path = image_path[:-4] + "_prediction_annotation" + str_end
	prediction_coordinates = draw_boxes(image, image_path, v_boxes, v_labels, v_scores, annotated_image_path)

	return prediction_coordinates, v_boxes, v_labels, v_scores, box_classes_scores



# image_path = "person_019.jpg"

# # Generate preiction
# YoloPredict_fromPath(image_path, 416, 416)