import numpy as np
from numpy import expand_dims
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import cv2

class BoundBox:
	def __init__(self, xmin, ymin, xmax, ymax, objness = None, classes = None):
		self.xmin = xmin
		self.ymin = ymin
		self.xmax = xmax
		self.ymax = ymax
		self.objness = objness
		self.classes = classes
		self.label = -1
		self.score = -1

	def get_label(self):
		if self.label == -1:
			self.label = np.argmax(self.classes)

		return self.label

	def get_score(self):
		if self.score == -1:
			self.score = self.classes[self.get_label()]

		return self.score

def _sigmoid(x):
	return 1. / (1. + np.exp(-x))

def decode_netout(netout, anchors, obj_thresh, net_h, net_w):
	grid_h, grid_w = netout.shape[:2]
	nb_box = 3
	netout = netout.reshape((grid_h, grid_w, nb_box, -1))
	nb_class = netout.shape[-1] - 5
	boxes = []
	netout[..., :2]  = _sigmoid(netout[..., :2])
	netout[..., 4:]  = _sigmoid(netout[..., 4:])
	netout[..., 5:]  = netout[..., 4][..., np.newaxis] * netout[..., 5:]
	netout[..., 5:] *= netout[..., 5:] > obj_thresh
 
	for i in range(grid_h*grid_w):
		row = i / grid_w
		col = i % grid_w
		for b in range(nb_box):
			# 4th element is objectness score
			objectness = netout[int(row)][int(col)][b][4]
			if(objectness.all() <= obj_thresh): continue
			# first 4 elements are x, y, w, and h
			x, y, w, h = netout[int(row)][int(col)][b][:4]
			x = (col + x) / grid_w # center position, unit: image width
			y = (row + y) / grid_h # center position, unit: image height
			w = anchors[2 * b + 0] * np.exp(w) / net_w # unit: image width
			h = anchors[2 * b + 1] * np.exp(h) / net_h # unit: image height
			# last elements are class probabilities
			classes = netout[int(row)][col][b][5:]
			box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, objectness, classes)
			boxes.append(box)
	return boxes
 
def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):
	new_w, new_h = net_w, net_h
	for i in range(len(boxes)):
		x_offset, x_scale = (net_w - new_w)/2./net_w, float(new_w)/net_w
		y_offset, y_scale = (net_h - new_h)/2./net_h, float(new_h)/net_h
		boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
		boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
		boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
		boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)
 
def _interval_overlap(interval_a, interval_b):
	x1, x2 = interval_a
	x3, x4 = interval_b
	if x3 < x1:
		if x4 < x1:
			return 0
		else:
			return min(x2,x4) - x1
	else:
		if x2 < x3:
			 return 0
		else:
			return min(x2,x4) - x3
 
def bbox_iou(box1, box2):
	intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
	intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])
	intersect = intersect_w * intersect_h
	w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin
	w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin
	union = w1*h1 + w2*h2 - intersect
	return float(intersect) / union
 
def do_nms(boxes, nms_thresh):
	if len(boxes) > 0:
		nb_class = len(boxes[0].classes)
		
		for c in range(nb_class):
			sorted_indices = np.argsort([-box.classes[c] for box in boxes])
			for i in range(len(sorted_indices)):
				index_i = sorted_indices[i]
				if boxes[index_i].classes[c] == 0: continue
				for j in range(i+1, len(sorted_indices)):
					index_j = sorted_indices[j]
					if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
						boxes[index_j].classes[c] = 0
	else:
		return
	
 
# load and prepare an image
def load_image_pixels(filename, shape):
	# load the image to get its shape
	image = load_img(filename)
	width, height = image.size
	# load the image with the required size
	image = load_img(filename, target_size=shape)
	# convert to numpy array
	image = img_to_array(image)
	# scale pixel values to [0, 1]
	image = image.astype('float32')
	image /= 255.0
	# add a dimension so that we have one sample
	image = expand_dims(image, 0)
	return image, width, height
 
# get all of the results above a threshold
def get_boxes(boxes, labels, thresh):
	v_boxes, v_labels, v_scores = list(), list(), list()
	box_classes_scores = list() #AG added this
	# enumerate all boxes
	for box in boxes:
		# enumerate all possible labels
		for i in range(len(labels)):
			# check if the threshold for this label is high enough
			if box.classes[i] > thresh:
				# print("box.classes = ", box.classes)
				# input("press enter")
				v_boxes.append(box)
				v_labels.append(labels[i])
				v_scores.append(box.classes[i]*100)
				box_classes_scores.append(box.classes)
				# don't break, many labels may trigger for one box
	box_classes_scores = np.array(box_classes_scores)
	# print("box_classes_scores = ", box_classes_scores)
	# print("box_classes_scores[0] = ", box_classes_scores[0])
	# input("press enter")
	return v_boxes, v_labels, v_scores, box_classes_scores
 
# draw all results
def draw_boxes(image, image_path, v_boxes, v_labels, v_scores, name_prediction_file):#filename, v_boxes, v_labels, v_scores):
	# load the image
	# data = plt.imread(filename)

	# plot the image
	# plt.clf()
	# plt.imshow(image.reshape(416, 416,3))
	image = cv2.imread(image_path)
	annotated_image = image.copy()

	
	# # get the context for drawing boxes
	# ax = plt.gca()
	# plot each box
	prediction_coordinates = []
	for i in range(len(v_boxes)):
		box = v_boxes[i]
		# get coordinates
		y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
		left, top, right, bottom = x1, y1, x2, y2

		# Append it to prediction_coordinates
		prediction_coordinates.append([left, top, right, bottom])



		# # calculate width and height of the box
		# width, height = x2 - x1, y2 - y1
		# # create the shape
		# rect = Rectangle((x1, y1), width, height, fill=False, color='white')
		# # draw the box
		# ax.add_patch(rect)
		# # draw text and score in top left corner
		label = "%s (%.3f)" % (v_labels[i], v_scores[i])
		# plt.text(x1, y1, label, color='white')
		# Draw rectangle 
		cv2.rectangle(annotated_image, (int(left), int(top)), (int(right), int(bottom)), (0, 0, 230), thickness=2)
		font                   = cv2.FONT_HERSHEY_SIMPLEX
		fontScale              = 1
		fontColor              = (0,0,255)
		thickness              = 2
		lineType               = 2

		cv2.putText(annotated_image,label, 
		    (x1,y1), 
		    font, 
		    fontScale,
		    fontColor,
		    thickness,
		    lineType)
	# show the plot
	# plt.show()
	# plt.grid(False)
	# plt.savefig(name_prediction_file)
	cv2.imwrite(name_prediction_file, annotated_image)#cv2.flip(annotated_image, 1))
	return prediction_coordinates

# def prediction_for_deepcover(model, photo_filename, input_w, input_h):

# 	# define the labels
# 	labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
# 		"boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
# 		"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
# 		"backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
# 		"sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
# 		"tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
# 		"apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
# 		"chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
# 		"remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
# 		"book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
	
# 	# define the anchors
# 	anchors = [[116,90, 156,198, 373,326], [30,61, 62,45, 59,119], [10,13, 16,30, 33,23]]

# 	class_threshold = 0.6

# 	# load and prepare image
# 	image, image_w, image_h = load_image_pixels(photo_filename, (input_w, input_h))

# 	# make prediction
# 	yhat = model.predict(image)

# 	# summarize the shape of the list of arrays
# 	print([a.shape for a in yhat])

# 	boxes = list()
# 	for i in range(len(yhat)):
# 		# decode the output of the network
# 		boxes += decode_netout(yhat[i][0], anchors[i], class_threshold, input_h, input_w)

# 	# correct the sizes of the bounding boxes for the shape of the image
# 	correct_yolo_boxes(boxes, image_h, image_w, input_h, input_w)

# 	# suppress non-maximal boxes
# 	do_nms(boxes, 0.5)

# 	# get the details of the detected objects
# 	v_boxes, v_labels, v_scores, box_classes_scores = get_boxes(boxes, labels, class_threshold)

# 	# summarize what we found
# 	for i in range(len(v_boxes)):
# 		print(v_labels[i], v_scores[i])

# 	# draw what we found
# 	draw_boxes(photo_filename, v_boxes, v_labels, v_scores)

# 	return box_classes_scores

def yolo_to_deepcover(yolo_prediction, input_h, input_w, image):

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

	# define the anchors
	anchors = [[116,90, 156,198, 373,326], [30,61, 62,45, 59,119], [10,13, 16,30, 33,23]]

	class_threshold = 0.6

	# input_w, input_h = 416, 416

	boxes = list()
	for i in range(len(yolo_prediction)):
		# decode the output of the network
		boxes += decode_netout(yolo_prediction[i][0], anchors[i], class_threshold, input_h, input_w)
	
	# correct the sizes of the bounding boxes for the shape of the image
	correct_yolo_boxes(boxes, input_h, input_w, input_h, input_w)

	# suppress non-maximal boxes
	do_nms(boxes, 0.5) # NEED to return this to work.

	# get the details of the detected objects
	v_boxes, v_labels, v_scores, box_classes_scores = get_boxes(boxes, labels, class_threshold)
	print("box_classes_scores = ",box_classes_scores)
	print("len(box_classes_scores) = ",len(box_classes_scores))
	# input("wait, press enter to contiu")
	name_prediction_file = "temp_image_prediction.png"
	draw_boxes(image, v_boxes, v_labels, v_scores, name_prediction_file)
	print("box_classes_scores = ", box_classes_scores)
	# input("Enter temp image display wait")
	if len(box_classes_scores) == 0:
		deepcover_input	= None
	else:
		deepcover_input = box_classes_scores[0].reshape(1,len(box_classes_scores[0])) # This only takes the first identified object in the list of objects detected. 
	
	return deepcover_input



