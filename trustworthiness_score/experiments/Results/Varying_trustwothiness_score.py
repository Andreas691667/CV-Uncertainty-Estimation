import os
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd

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


def IoU_calculator(left1, top1, right1, bottom1,
                    left2, top2, right2, bottom2):


   square1_area      = SquareAreaCalculator(left1, top1, right1, bottom1)
   square2_area      = SquareAreaCalculator(left2, top2, right2, bottom2)

   left3, top3, right3, bottom3 = IntersectingRectangle(left1, top1, right1, bottom1,
                                                        left2, top2, right2, bottom2)
    
   if left3 == None:
      IoU = 0     
   else:
      intersection_area = SquareAreaCalculator(left3, top3, right3, bottom3)
      union_area        = square1_area + square2_area - intersection_area
      IoU = (intersection_area) / (union_area)  # We smooth our devision to avoid 0/0

   return IoU, left3, top3, right3, bottom3


class image_summary_class():
   def __init__(self, image_name, image_id):
      self.image_name = image_name 
      self.image_id = image_id

      self.array_of_ground_truth_predictions = []
      self.array_of_predictions = []
      self.array_of_features = []

      self.ground_turth_TP = None
      self.ground_turth_FP = None
      self.ground_turth_FN = None

      self.features_TP = None
      self.features_FP = None
      self.features_FN = None

      self.frame_trustworthiness_score = None


class person():
   def __init__(self, person_id, left, top, right, bottom):
      self.id  = person_id

      self.left   = left
      self.top    = top
      self.right  = right
      self.bottom = bottom

      self.found_overlapping_features_flag = False
      self.list_of_overlapping_features = []
      self.list_of_overlapping_features_intersection_areas = []
      self.list_of_overlapping_features_id = []
      self.list_of_overlapping_features_type_id = []
      self.prediction_trustworthiness_score = 0

      self.P_flag = None # flag showing this is a prediction 
      self.G_flag = None # flag showing that a ground truth agrees with prediciton if P_flag == True. If P_flag == False then G_flag indicates that this is person is generated from an annotation.


class feature_specification():
    def __init__(self,feature_id, feature_type_id, left, top, right, bottom):
        self.id      = feature_id
        self.type_id = feature_type_id
        self.left = left
        self.top  = top
        self.right = right
        self.bottom = bottom

        if feature_type_id == 0:
            # This is a facial feature
            self.beta = 1

        elif feature_type_id == 1:
            # This is a palm feature
            self.beta = 1

        elif feature_type_id == 2:
            # This is a legs feature
            self.beta = 1
        
        self.assigned_person_id = None
        self.found_overlapping_prediction_flag = False

# ===========================================
# == Read in file
# ===========================================
f_n = 19#20#19

file_path_annotations = "Experiment"+str(f_n)+"/LogsAnnotations"+str(f_n)+".csv"
file_path_predictions = "Experiment"+str(f_n)+"/LogsPredictions"+str(f_n)+".csv"
file_path_features    = "Experiment"+str(f_n)+"/LogsFeatures"+str(f_n)+".csv"

df_annotations = pd.read_csv(file_path_annotations)
df_predictions = pd.read_csv(file_path_predictions)
df_features    = pd.read_csv(file_path_features)

# print(df_annotations)

# =============================================
# == Loop over images
# =============================================
print("max(df_annotations['ImageID']) = ", max(df_annotations["ImageID"]))


log_file_path = "results_summary_"+str(f_n)+"_4_face_palm_legs.csv"
with open(log_file_path, 'w') as log_file: 
	log_file.write('R_i_min,beta_legs,model_confidence_threshold,TCS_threshold,TP,FP,FN\n')

experiment_counter = 0

for beta_legs in [1]:#[0.5,0.75,1]: 
	for model_confidence_threshold in [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]: 
		# for TCS_i_h_threshold in [1,10,20,30,40,50,60,70,75,80,85,90, 100, 200, 700]:#, 1000, 1200, 1500, 1700, 2000]: # INRIA
		for TCS_i_h_threshold in [1,50,80,85, 100, 200, 700]:#, 1000, 1200, 1500, 1700, 2000]:   # COCO	
			for R_i_min in [70]:
				
				experiment_counter += 1
				print("Experiments progresion = "+ str(experiment_counter) + "/" + str(7*21))

				df_predictions_filtered = df_predictions[df_predictions.Prediction_confidence >= model_confidence_threshold]
				print(df_predictions_filtered)
				
				TP_G_total   = 0
				FP_G_total   = 0
				FN_G_total   = 0

				for image_id in range(max(df_annotations["ImageID"])):
					image_id += 1

					idx = df_annotations[df_annotations["ImageID"]==image_id].index.values[0]
					image_name = df_annotations["ImageName"][idx]
					
					# print("==============================")
					# print("image_name = ",image_name)  
					# print("==============================")
					image_summary = image_summary_class(image_name, image_id)

					# =================================================================================
					# == Loop over Predictions
					# =================================================================================
	
					TP_G                = 0 
					FP_G                = 0 
					FN_G                = 0 

					IoU_threshold = 0.7#0.4#0.7

					num_of_annotations = len(df_annotations[df_annotations["ImageID"]==image_id].index.values)
					num_of_features    = len(df_features[df_features["ImageID"]==image_id].index.values)
					num_of_predictions = len(df_predictions_filtered[df_predictions_filtered["ImageID"]==image_id].index.values)

					# Check if features speicifaitons were detected and set a flag for that.
					features_detected_flag = False
					if num_of_features >= 1:
						features_detected_flag = True

					# Loop over predictions
					for pred_idx in df_predictions_filtered[df_predictions_filtered["ImageID"]==image_id].index.values:
						
						G_flag = False
						# F_falg = False
						
						# Extract prediction box bouandaries
						prediction_left   = df_predictions_filtered["Prediction_left"][pred_idx] 
						prediction_top    = df_predictions_filtered["Prediction_top"][pred_idx] 
						prediction_right  = df_predictions_filtered["Prediction_right"][pred_idx] 
						prediction_bottom = df_predictions_filtered["Prediction_bottom"][pred_idx] 
						prediction_confidence = df_predictions_filtered["Prediction_confidence"][pred_idx] 
						
						# =================================================================================
						# == Calculate Trustworthiness Score on Prediction
						# =================================================================================

						# Reset trustworthiness_score for predcition.
						TCS_i_h = 0

						if features_detected_flag == True: 

							# Loop over features detection
							features_counter = 0
							for feature_idx in df_features[df_features["ImageID"]==image_id].index.values:

								# Extract fetures box bouandaries
								feature_type_id = df_features["Feature_TypeID"][feature_idx] 
								feature_left    = df_features["Feature_left"][feature_idx] 
								feature_top     = df_features["Feature_top"][feature_idx] 
								feature_right   = df_features["Feature_right"][feature_idx] 
								feature_bottom  = df_features["Feature_bottom"][feature_idx] 

								# Find the rectangle coordinates of the overlap between feature and prediction
								I_left, I_top, I_right, I_bottom = IntersectingRectangle(feature_left, feature_top, feature_right, feature_bottom,
																						prediction_left, prediction_top, prediction_right, prediction_bottom)
							
								# R_i is the percentage of pixels for F_z that overlap with prediciton C_h.
								intersection_area = SquareAreaCalculator(I_left, I_top, I_right, I_bottom)
								feature_area      = SquareAreaCalculator(feature_left, feature_top, feature_right, feature_bottom)

								if feature_type_id   == 0:
									# This is a facial feature
									beta = 1
								elif feature_type_id == 1:
									# This is a palm feature
									beta = 1
								elif feature_type_id == 2:
									# This is a legs feature
									beta = 1#beta_legs

								if intersection_area == None:
									R_i = 0
								else:
									R_i = 100 * intersection_area / feature_area
						
								# If overlap 
								if (I_top is not None) and (R_i >= R_i_min):
									a_i = 1

									# Calculate prediciton trustworthiness score TCS_i_h based on features found.
									TCS_i_h  += beta * a_i * intersection_area
							

						# if Trustworhtiness score is more than threshold then run ground truth assessment on Predciiton below
						if TCS_i_h > TCS_i_h_threshold:

							# =================================================================================
							# == Ground truth assesment on Prediction
							# =================================================================================
							# Loop over grouth truth annotaitons
							for annot_idx in df_annotations[df_annotations["ImageID"]==image_id].index.values:
								
								# Extract annotation box bouandaries
								annotation_left   = df_annotations["Annotation_left"][annot_idx] 
								annotation_top    = df_annotations["Annotation_top"][annot_idx] 
								annotation_right  = df_annotations["Annotation_right"][annot_idx] 
								annotation_bottom = df_annotations["Annotation_bottom"][annot_idx] 

								# Calculate Intersection over Union (IoU)
								IoU, I_left, I_top, I_right, I_bottom =  IoU_calculator(prediction_left, prediction_top, prediction_right, prediction_bottom,
																						annotation_left, annotation_top, annotation_right, annotation_bottom)
								# print("IoU = ", IoU)
								if IoU >= IoU_threshold:
									G_flag = True 
									break

							if G_flag: #and prediction_confidence >= model_confidence_threshold:
								TP_G += 1
							else:
								FP_G += 1


					# =================================================================================
					# == Calculate FN based on groudntruth
					# =================================================================================
					TP_G_total                += TP_G
					FP_G_total                += FP_G
					FN_G_total                += num_of_annotations - TP_G 


				# =================================================================================
				# == Precision, Recall, F1 Score Calculations
				# =================================================================================

				Precision_G = TP_G_total / (TP_G_total+FP_G_total)
				Recall_G   = TP_G_total /(TP_G_total + FN_G_total)
				F_Measure_G = (2 * Precision_G * Recall_G) / (Precision_G + Recall_G)

	
				print("==")

				with open(log_file_path, 'a') as log_file: 
					log_file.write('%.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f\n' %\
					(R_i_min, beta_legs, model_confidence_threshold,TCS_i_h_threshold, TP_G_total, FP_G_total, FN_G_total))


# ======================================================
# = Figure summarising sum of TP,FP, FN
# ======================================================
# SMALL_SIZE = 8
# MEDIUM_SIZE = 12
# BIGGER_SIZE = 12

# plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
# plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
# plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
# plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# fig = plt.figure()
# ax = fig.add_axes([0.09,0.06,0.90,0.92])
# X = np.arange(3)

# ax.bar(X - 0, [TS_F1_Score[0], TS_F1_Score[1], TS_F1_Score[2]], color = 'black', width = 0.35, label='Trustworhtiness in Classification')

# # Plt ground truth line
# x = np.linspace(-1,5,100)
# y = GT_F1_Score*np.ones(x.shape)
# plt.plot(x, y, '--r', linewidth='2',label='Ground Truth')

# ax.patch.set_edgecolor('black')  
# ax.patch.set_linewidth('1') 

# plt.grid(False)

# ax.legend()
# plt.xticks([0,1,2], ['Face', 'Face/Hand','Face/Hand/Legs'], rotation=0)
# # if df["ExperimentID"][0] > 7:
# #     plt.ylim([0, 4])
# # else:
# plt.xlim([-0.4, 2.4])
# plt.ylim([0, 1])
# plt.ylabel("F1 Score") 
# plt.savefig('Varying_Features.pdf')
