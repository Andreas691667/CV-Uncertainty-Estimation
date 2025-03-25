import matplotlib.pyplot as plt
import imutils
import cv2
from yolo.get_yolo_prediction import *

def IntersectingRectangle(left1, top1, right1, bottom1,
                          left2, top2, right2, bottom2):
 
    # gives bottom-left point of intersection rectangle
    left3 = max(left1, left2)
    bottom3 = min(bottom1, bottom2)
 
    # gives top-right point of intersection rectangle
    right3 = min(right1, right2)
    top3 = max(top1, top2)
 
    # no intersection
    if (left3 > right3 or bottom3 < top3) :
        # print("No intersection")
        left3   = None
        top3    = None
        right3  = None
        bottom3 = None

    return left3, top3, right3, bottom3
 
def SquareAreaCalculator(left, top, right, bottom):
   if left == None:
   	return None
   else:
   	return abs((right-left)*(top-bottom))

def CalculateTrustworthiness(image, image_cv2, image_summary):

	frame_id = image_summary.image_id
	
	# Initialise Trustwothiness score for whole image
	TCS_i = float('inf')
	
	# Initialise True-Positive (TP), False-Positive Or  False-Negative (FPFN)
	TP = 0
	FP = 0
	FN = 0

	R_i_min = 10 # Minimum threshold for considering and overlap.

	# Check if features speicifaitons were detected and set a flag for that.
	features_detected_flag = False
	if len(image_summary.array_of_features) >= 1:
		features_detected_flag = True

	# Check if predcitions were detected and set a flag for that.
	predictions_detected_flag = False
	if len(image_summary.array_of_predictions) >= 1:
		predictions_detected_flag = True

	# If pedictions existis but not featrues speiciications or vice versa Trustwothiness score = 0.
	if predictions_detected_flag == False and features_detected_flag == True: 
		image_summary.frame_trustworthiness_score = 0
		FN = len(image_summary.array_of_features)


	elif predictions_detected_flag == True and features_detected_flag == False: 
		image_summary.frame_trustworthiness_score = 0
		FP = len(image_summary.array_of_predictions)


	# If predicitons and features do not exist. Make sure that the input data is similar to training data using dissimilarity measures.
	elif predictions_detected_flag == False and features_detected_flag == False:

		pass_dissimlarity_flag = True
		if pass_dissimlarity_flag == True:
			# If it passes dissimlarity measures test then trustworthiness_score is high (100%) that nothing is in the frame.
			image_summary.frame_trustworthiness_score = 100
		else:
			image_summary.frame_trustworthiness_score =  0

	
	# If prediciiton and features specificaiton exists then calcuate trustworthiness:
	elif predictions_detected_flag == True and features_detected_flag == True: 
		# print("FOUND PREDICTIONS AND FEATURES: Caluclating TCS ...")
		# Loop over predictions
		for C_h in image_summary.array_of_predictions:
			# print("=======================================================")
			# Reset trustworthiness_score for predcition
			TCS_i_h = 0
			# Loop over Features
			for F_z in image_summary.array_of_features:

				# Find the rectangle coordinates of the overlap between feature and prediction
				I_left, I_top, I_right, I_bottom = IntersectingRectangle(F_z.left, F_z.top, F_z.right, F_z.bottom,
																		C_h.left, C_h.top, C_h.right, C_h.bottom)

				Overlap_C_h_and_F_z_pixels = image.reshape(416, 416, 3)
				Overlap_C_h_and_F_z_pixels = Overlap_C_h_and_F_z_pixels[I_top:I_bottom, I_left:I_right,:]

				# R_i is the percentage of pixels for F_z that overlap with prediciton C_h.
				intersection_area = SquareAreaCalculator(I_left, I_top, I_right, I_bottom)
				feature_area      = SquareAreaCalculator(F_z.left, F_z.top, F_z.right, F_z.bottom)
				if intersection_area == None:
					R_i = 0
				else:
					R_i = 100 * intersection_area /feature_area

				# If overlap 
				if (I_top is not None) and (R_i>= R_i_min):
					a_i = 1
					# Append to features list in predcition class
					C_h.list_of_overlapping_features.append(F_z)

					C_h.list_of_overlapping_features_intersection_areas.append(intersection_area)

					C_h.list_of_overlapping_features_type_id.append(F_z.type_id)

					C_h.list_of_overlapping_features_id.append(F_z.id)

					# Set found overlapping feature flag in prediction class to True
					C_h.found_overlapping_features_flag =  True

					# Set found overlapping prediction flag in feature class to True
					F_z.found_overlapping_prediction_flag = True

					# Cacluate prediciton trustworthiness score TCS_i_h based on features found.
					TCS_i_h  += F_z.beta * a_i * intersection_area

			
			TCS_i_h_threshold = 1
			if TCS_i_h > TCS_i_h_threshold:
				TP+=1
			else:
				FP+=1

			C_h.prediction_trustworthiness_score = TCS_i_h


			# Caclaute overall trustworthiness score TCS_i for frame (image). Min(TCS_i_1, TCS_i_2,... TCS_i_H)
			TCS_i = min(TCS_i,TCS_i_h)
			# print("TCS_i = ",TCS_i)


		image_summary.frame_trustworthiness_score = TCS_i

		# Loop over Features
		for F_z in image_summary.array_of_features:
			if F_z.found_overlapping_prediction_flag == False:
				FN += 1

	print("feature_TP = ",TP)
	print("feature_FP = ",FP)
	print("feature__FN = ",FN)

	image_summary.features_TP = TP
	image_summary.features_FP = FP
	image_summary.features_FN = FN