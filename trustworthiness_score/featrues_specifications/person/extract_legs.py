import torch
import os

import cv2
from feature_definition import feature_specification

def ExtractLegs(image, annotated_image, image_name, results_dir_name, image_summary):

	# weights_file = "runs/train/yolo_legs7/weights/best.pt"
	weights_file = "../featrues_specifications/person/yolov5/runs/train/yolo_legs7/weights/best.pt"
	model = torch.hub.load('../featrues_specifications/person/yolov5', 'custom', path=weights_file, source='local') 

	# Images
	# img = './crop001512.png'  # or file, Path, PIL, OpenCV, numpy, list

	# print(img)
	# Inference
	results = model(image)

	# Results
	# results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
	# print(results.pandas().xyxy[0])
	# results.save()
	# input("kill script here")

	# print("Resecults = ", results)
	# input("Kill script here")
	boxes_pandas = results.pandas().xyxy[0]

	confidence_threshold = 0.4 # Threshold is  found from F1 score curve. Best F1 score at 0.384 
	for i in range(len(boxes_pandas)):
		if boxes_pandas.confidence[i] >= confidence_threshold:
			left = int(boxes_pandas.xmin[i])
			top  = int(boxes_pandas.ymin[i])
			right = int(boxes_pandas.xmax[i])
			bottom = int(boxes_pandas.ymax[i])


			# features_coordinates.append([left, top, right, bottom])
			image_summary.array_of_features.append(feature_specification(None, 2, left, top, right, bottom))

			# draw the bounding box on our image
			cv2.rectangle(annotated_image, (int(left), int(top)), (int(right), int(bottom)), (230, 0, 230), thickness=2)

	# annotated_image_path = results_dir_name+"06/"+image_name
	# cv2.imwrite(annotated_image_path, annotated_image)#cv2.flip(annotated_image, 1))

	
