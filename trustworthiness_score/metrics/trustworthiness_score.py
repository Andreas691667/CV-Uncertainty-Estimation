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
		# print("NO PREDICTIONS FOUND: TCS = 0.")

	elif predictions_detected_flag == True and features_detected_flag == False: 
		image_summary.frame_trustworthiness_score = 0
		FP = len(image_summary.array_of_predictions)
		# print("NO FEATURES FOUND: TCS = 0.")

	# If predicitons and features do not exist. Make sure that the input data is similar to training data using dissimilarity measures.
	elif predictions_detected_flag == False and features_detected_flag == False:
		# print("No available data to calculate trustworthiness. Check dissmilarity measure.")
		# print("Make sure that the input data is similar to training data using dissimilarity measures.")
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
				
				# # Count the number of pixels in the overlap
				# F_z_pixels = image.reshape(416, 416, 3)
				# F_z_pixels = F_z_pixels[I_top:I_bottom, I_left:I_right,:]
				# boolArr_not_masked = (F_z_pixels[:,:,0] != 0) | (F_z_pixels[:,:,1] != 0) | (F_z_pixels[:,:,2] != 0)
				# num_pixels_F_z = boolArr_not_masked.sum() # Number of pixels in feature detected (F_z)

				Overlap_C_h_and_F_z_pixels = image.reshape(416, 416, 3)
				Overlap_C_h_and_F_z_pixels = Overlap_C_h_and_F_z_pixels[I_top:I_bottom, I_left:I_right,:]
				# boolArr_not_masked = (Overlap_C_h_and_F_z_pixels[:,:,0] != 0) | (Overlap_C_h_and_F_z_pixels[:,:,1] != 0) | (Overlap_C_h_and_F_z_pixels[:,:,2] != 0)
				# num_pixels_overlapping = boolArr_not_masked.sum() # Number of shared pixels between feature (F_z) and prediciton (C_h)

				# R_i = 100*(num_pixels_overlapping/(num_pixels_F_z)) # R_i is the percentage of pixels for F_z that overlap with prediciton C_h.
				# print("R_i_pixels = ",R_i)

				# R_i is the percentage of pixels for F_z that overlap with prediciton C_h.
				intersection_area = SquareAreaCalculator(I_left, I_top, I_right, I_bottom)
				feature_area      = SquareAreaCalculator(F_z.left, F_z.top, F_z.right, F_z.bottom)
				if intersection_area == None:
					R_i = 0
				else:
					R_i = 100 * intersection_area /feature_area
				# print("R_i_area = ",R_i)

				# input("Kill script here!")
				
				# print("num_pixels_F_z =", num_pixels_F_z)
				# print("num_pixels_overlapping =", num_pixels_overlapping)
				
				
				# If overlap 
				if (I_top is not None) and (R_i>= R_i_min):
					a_i = 1
					# Append to features list in predcition class
					C_h.list_of_overlapping_features.append(F_z)

					# Set found overlapping feature flag in prediction class to True
					C_h.found_overlapping_features_flag =  True

					# Set found overlapping prediction flag in feature class to True
					F_z.found_overlapping_prediction_flag = True

					# Cacluate prediciton trustworthiness score TCS_i_h based on features found.
					TCS_i_h  += F_z.beta * a_i * intersection_area

					# print("TCS_i_h = ", TCS_i_h)
					# plt.clf()
					# plt.imshow(Overlap_C_h_and_F_z_pixels)
					# plt.grid(False)
					# plt.savefig("100_feature_cropped.png")

			
			TCS_i_h_threshold = 100
			if TCS_i_h > TCS_i_h_threshold:
				TP+=1
			else:
				FP+=1

			C_h.prediction_trustworthiness_score = TCS_i_h


			# Caclaute overall trustworthiness score TCS_i for frame (image). Min(TCS_i_1, TCS_i_2,... TCS_i_H)
			TCS_i = min(TCS_i,TCS_i_h)
			# print("TCS_i = ",TCS_i)


		image_summary.frame_trustworthiness_score = TCS_i

		# Count TP and FP
		# Loop over predictions
		# for C_h in image_summary.array_of_predictions:
			# if feature found flag == True: 
			# TP += 1
		# else:
			# it could be a FP or TP 

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
		
		
	# 	# Loop over Features 
	# 		# Check that feature found an overlapping predcition
	# 		# If found overlapping prediciton flag is False
	# 			# TCS_i = 0
	# 			# break





	# 			# R_i = 100*(num_non_masked/(num_non_masked + num_masked)) # R_i is the percentage of pixels for F_i that are not masked.
	# 			# R_i_array.append(R_i)

	# 			# if R_i >= R_i_min:
	# 			# 	a_i = 1
	# 			# 	detected_features_counter += 1
	# 			# else:
	# 			# 	a_i = 0

	# 			# a_i_array.append(a_i)  # where a_i = 1 for R_i >= R_i_min, otherwise a_i = 0

	# 			# prediction_trustworthiness_score += array_of_beta[i] * a_i * R_i





	# # print("prediction_image = ", prediction_coordinates)
	# # print("explained_image = ", explained_image)
	# # print("array_of_features = ", array_of_features)
	# # print("array_of_beta = ", array_of_beta)

	# # image_cv2 = cv2.imread(image_path)
	# # image_cv2 = imutils.resize(image_cv2, width=500)

	# # load and prepare image
	# # input_w, input_h = 416, 416
	# # image, _, _ = load_image_pixels(image_path, (416, 416))

	# print("prediction_coordinates = ",prediction_coordinates)
	# print("prediction_coordinates = ",prediction_coordinates)



	# i             = 0   # Features ID for feature F_i.
	# R_i_array     = []  # Array for R_i calculation.
	# R_i_min       = 10  # minimum percentage of non masked pixels required in a detected feature in order for the main classifier classification and the feature detected to considered as agreeing.
	
	# a_i_array = [] # array for a_i storage.

	# detected_features_counter = 0

	# trustworthiness_score = 0
	# print("array_of_features = ", array_of_features)
	# for F_i in array_of_features:
	# 	print("F_i = ", F_i)
	# 	print("len(F_i) = ", len(F_i))

	# 	if len(F_i) != 0 :

	# 		# testing transfer
	# 		left   = F_i[0][0]
	# 		right  = F_i[0][2]
	# 		top    = F_i[0][1]
	# 		bottom = F_i[0][3]

	# 		feature_explained_image = explained_image.reshape(416, 416, 3)
	# 		feature_explained_image = feature_explained_image[top:bottom, left:right,:]
	# 		# print(explained_image.shape)
	# 		plt.imshow(feature_explained_image)
	# 		plt.grid(False)
	# 		plt.savefig("06_feature_explained_cropped_"+str(i+1)+".png")
	# 		print("feature_explained_image = ",feature_explained_image)

	# 		# boolArr_not_masked = (explained_image[:,feature[0][0]:feature[0][2],feature[0][1]:feature[0][3],0] != 1) | \
	# 		# 					 (explained_image[:,feature[0][0]:feature[0][2],feature[0][1]:feature[0][3],1] != 1) | \
	# 		# 					 (explained_image[:,feature[0][0]:feature[0][2],feature[0][1]:feature[0][3],2] != 1)

	# 		# Find which pixels of the explanation are masked
	# 		boolArr_not_masked = (feature_explained_image[:,:,0] != 0) | (feature_explained_image[:,:,1] != 0) | (feature_explained_image[:,:,2] != 0)
	# 		boolArr_masked     = (feature_explained_image[:,:,0] == 0) & (feature_explained_image[:,:,1] == 0) & (feature_explained_image[:,:,2] == 0)

	# 		print("boolArr_not_masked = ",boolArr_not_masked)
	# 		print("boolArr_masked = ",boolArr_masked)

	# 		print("boolArr_not_masked.sum() = ",boolArr_not_masked.sum())
	# 		print("boolArr_masked.sum() = ",boolArr_masked.sum())

	# 		num_non_masked = boolArr_not_masked.sum() # Number of non masked pixels
	# 		num_masked = boolArr_masked.sum()         # Number of masked pixels

	# 		R_i = 100*(num_non_masked/(num_non_masked + num_masked)) # R_i is the percentage of pixels for F_i that are not masked.
	# 		R_i_array.append(R_i)

	# 		if R_i >= R_i_min:
	# 			a_i = 1
	# 			detected_features_counter += 1
	# 		else:
	# 			a_i = 0

	# 		a_i_array.append(a_i)  # where a_i = 1 for R_i >= R_i_min, otherwise a_i = 0

	# 		trustworthiness_score += array_of_beta[i] * a_i * R_i

	# 		i += 1

	# 	else:
	# 		print("Feature not detected")
	# 		i += 1
	# 		# del array_of_features[i]
	# 		# del array_of_beta[i]


	# if detected_features_counter == 0:
	# 	trustworthiness_score = 0
	# else:
	# 	trustworthiness_score = trustworthiness_score/detected_features_counter
	# print("trustworthiness_score = ",trustworthiness_score)
	# 		# print("np.array(boolArr_not_masked).sum() =",np.array(boolArr_not_masked).sum())



	# 		# print(boolarr.sum())


	# 		# plt.clf()
	# 		# face_extraction_image = image[0]
	# 		# plt.imshow(face_extraction_image[test_y1:test_y2, test_x1:test_x2,:])
	# 		# plt.plot([test_x1,test_x2],[test_y1,test_y2],color="black")
	# 		# plt.grid(False)
	# 		# plt.savefig('test_00_image.png')

	# 		# cv2.line(image_cv2, (int(test_x1), int(test_y1)), (int(test_x2), int(test_y2)), (0, 0, 230), thickness=2)
	# 		# cv2.imwrite("test_01_image_cv2.png", image_cv2)

			

	# 		# boolArr_not_masked = (explained_image[:,feature[0][0]:feature[0][2],feature[0][1]:feature[0][3],0] != 1) | \
	# 		# 					 (explained_image[:,feature[0][0]:feature[0][2],feature[0][1]:feature[0][3],1] != 1) | \
	# 		# 					 (explained_image[:,feature[0][0]:feature[0][2],feature[0][1]:feature[0][3],2] != 1)

	# 		# print("feature = ", feature)
	# 		# print("feature[0][0] = ", feature[0][0])
	# 		# print("feature[0][1] = ", feature[0][1])
	# 		# print("feature[0][2] = ", feature[0][2])
	# 		# print("feature[0][3] = ", feature[0][3])

	# 		# print("===============================================")
	# 		# print("Transferred readings")
	# 		# print("left = ", feature[0][0])
	# 		# print("top = ", feature[0][1])
	# 		# print("right = ", feature[0][2])
	# 		# print("bottom = ", feature[0][3])
	# 		# print("===============================================")

			

	# 		# temp = image
	# 		# temp = temp.reshape(416, 416, 3)
	# 		# plt.clf()
	# 		# plt.imshow(temp)
	# 		# plt.grid(False)
	# 		# plt.savefig("image_temp.png")

	# 		# annotated_image = image_cv2.copy()
	# 		# feature1 = feature[0][1]
	# 		# feature2 = feature[0][3]
	# 		# print("feature1 = ",feature1)
	# 		# print("feature2 = ",feature2)


	# 		# cv2.rectangle(annotated_image, (int(feature[0][0]), int(feature[0][1])), (int(feature[0][2]), int(feature[0][3])), (0, 0, 230), thickness=2)
	# 		# cv2.imwrite("06_face_cv2.png", annotated_image)

	# 		# image_cropped = image.copy()
	# 		# image_cropped = image_cropped.reshape(416, 416, 3)
	# 		# image_cropped = image_cropped[int(top):int(bottom),int(left):int(right),:]
	# 		# plt.clf()
	# 		# plt.imshow(image_cropped)
	# 		# plt.grid(False)
	# 		# plt.savefig("06_prediction_tensorflow.png")

	
			
			
	# 		# plt.clf()
	# 		# temp = explained_image
	# 		# explained_image = explained_image.reshape(416, 416, 3)
	# 		# explained_image = explained_image[feature[0][0]:feature[0][2],feature[0][1]:feature[0][2],:]
	# 		# # print(explained_image.shape)
	# 		# plt.imshow(explained_image)
	# 		# plt.grid(False)
	# 		# plt.savefig("07_feature_explained_cropped.png")
	# 		# cv2.rectangle(explained_image, (int(feature[0][0]), int(feature[0][1])), (int(feature[0][2]), int(feature[0][3])), (0, 0, 230), thickness=2)
	# 		# cv2.imwrite("2.png", explained_image)

	# 	# else:
	# 	# 	print("Feature not detected")
		

	# # Find which pixels of the explanation are masked
	# # boolArr_not_masked = (explained_image[:,:,:,0] != 1) | (explained_image[:,:,:,1] != 1) | (explained_image[:,:,:,2] != 1)
	# # boolArr_masked     = (explained_image[:,:,:,0] == 1) & (explained_image[:,:,:,1] == 1) & (explained_image[:,:,:,2] == 1)

	# # Extract which pixels overlap between not masked and explanation boundaries
	# # print("explained_image.shape() =",explained_image.shape)
	# # plt.clf()
	# # plt.imshow(explained_image.reshape(416, 416, 3))
	# # plt.grid(False)
	# # plt.savefig("explained_image.png")

