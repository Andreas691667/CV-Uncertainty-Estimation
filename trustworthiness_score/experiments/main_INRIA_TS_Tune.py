import os
import sys
sys.path.append('../models')
sys.path.append('../explanation_strategies')
sys.path.append('../featrues_specifications')
sys.path.append('../featrues_specifications/person/yolov5')
sys.path.append('../metrics')


from yolo.predict_yolo import YoloPredict, YoloPredict_fromPath
from generate_explanation import GenerateExplanation
from person.extract_palm import ExtractPalm
from person.extract_face import ExtractFace
from person.extract_legs import ExtractLegs
from trustworthiness_score_tune import CalculateTrustworthiness

from yolo.get_yolo_prediction import *

import pandas as pd


# =================================================================================
# == Results setup
# =================================================================================
ExperimentID = 20#17
min_class_threshold = 0.01 # define the probability threshold for detected objects

# trustworthiness_score_thresholds = [1,50,100,200,300,400,500,1000]
# beta_legs = [1,0.75,0.5,0.25,0.1]
Face_extraction = True
Palm_extraction = True
Legs_extraction = True
results_dir_name = "Results/Experiment" + str(ExperimentID) + "/"
if not os.path.exists(results_dir_name):
   os.makedirs(results_dir_name)
if not os.path.exists(results_dir_name+"00/"):
   os.makedirs(results_dir_name+"00/")
if not os.path.exists(results_dir_name+"01/"):
   os.makedirs(results_dir_name+"01/")
if not os.path.exists(results_dir_name+"02/"):
   os.makedirs(results_dir_name+"02/")
if not os.path.exists(results_dir_name+"03/"):
   os.makedirs(results_dir_name+"03/")
if not os.path.exists(results_dir_name+"04/"):
   os.makedirs(results_dir_name+"04/")
if not os.path.exists(results_dir_name+"05/"):
   os.makedirs(results_dir_name+"05/")
if not os.path.exists(results_dir_name+"06/"):
   os.makedirs(results_dir_name+"06/")
if not os.path.exists(results_dir_name+"07/"):
   os.makedirs(results_dir_name+"07/")


class log_summary():
   def __init__(self):
      self.ExperimentID         = []
      self.ImageName            = []
      self.ImageID              = []

      self.TCS_image            = []
      self.TP_ground            = []
      self.TP_features          = []
      self.FP_ground            = []
      self.FP_features          = [] 
      self.FN_ground            = []
      self.FN_features          = []

      self.PredictionID         = []
      self.TCS_prediction       = []

      self.FeatureID            = [] 
      self.Feature_TypeID       = []
      self.IntersectionArea     = []
   

   def append(self, ExperimentID, ImageName, ImageID, TCS_image, TP_ground, TP_features, FP_ground, FP_features, FN_ground, FN_features, PredictionID, TCS_prediction, FeatureID, Feature_TypeID, IntersectionArea):
      self.ExperimentID.append(ExperimentID)
      self.ImageName.append(ImageName) 
      self.ImageID.append(ImageID) 

      self.TCS_image.append(TCS_image)
      self.TP_ground.append(TP_ground)
      self.TP_features.append(TP_features)
      self.FP_ground.append(FP_ground)
      self.FP_features.append(FP_features) 
      self.FN_ground.append(FN_ground)
      self.FN_features.append(FN_features)

      self.PredictionID.append(PredictionID)         
      self.TCS_prediction.append(TCS_prediction)
      self.FeatureID.append(FeatureID)
      self.Feature_TypeID.append(Feature_TypeID)
      self.IntersectionArea.append(IntersectionArea)
      
      

   def write_file(self, output_folder, file_name):
      # Folder "results" if not already there
      # output_folder = "tests_logs"
      if not os.path.exists(output_folder):
         os.makedirs(output_folder)

      file_path = os.path.join(output_folder, file_name)
      with open(file_path, 'w') as log_file: 
         log_file.write('ExperimentID,ImageName,ImageID,TCS_image,TP_ground,TP_features,FP_ground,FP_features,FN_ground,FN_features,PredictionID,TCS_prediction,FeatureID,Feature_TypeID,IntersectionArea\n')
         for i in range(len(self.ExperimentID)):
            log_file.write('%d, %s, %d, %3.3f, %d, %d, %d, %d, %d, %d, %d, %3.3f, %d, %d, %d \n' %\
               (self.ExperimentID[i],self.ImageName[i], self.ImageID[i], self.TCS_image[i], self.TP_ground[i], self.TP_features[i], self.FP_ground[i], self.FP_features[i], self.FN_ground[i], self.FN_features[i], self.PredictionID[i], self.TCS_prediction[i], self.FeatureID[i], self.Feature_TypeID[i], self.IntersectionArea[i]))
      print('Log file SUCCESSFULLY generated!')


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
   return abs((right-left)*(top-bottom))

def IoU_calculator(left1, top1, right1, bottom1,
                    left2, top2, right2, bottom2):

   print("left1 = ", left1)
   print("top1 = ", top1)
   print("right1 = ", right1)
   print("bottom1 = ", bottom1)
   print("left2 = ", left2)
   print("top2 = ", top2)
   print("right2 = ", right2)
   print("bottom2 = ", bottom2)


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


class log_predictions_live():
   def __init__(self,output_folder, file_name):
      if not os.path.exists(output_folder):
         os.makedirs(output_folder)

      self.file_path = os.path.join(output_folder, file_name)
      if not os.path.exists(self.file_path):
         with open(self.file_path, 'w') as self.log_file: 
            self.log_file.write('ExperimentID,ImageName,ImageID,PredictionID,Prediction_left,Prediction_top,Prediction_right,Prediction_bottom,Prediction_confidence\n')
         
   def append(self, ExperimentID, ImageName, ImageID, PredictionID, Prediction_left, Prediction_top, Prediction_right, Prediction_bottom, Prediction_confidence):
      with open(self.file_path, 'a') as log_file: 
         log_file.write('%d, %s, %d, %d, %d, %d, %d, %d, %.1f\n' %\
         (ExperimentID, ImageName, ImageID, PredictionID, Prediction_left, Prediction_top, Prediction_right, Prediction_bottom,Prediction_confidence))


class log_annotations_live():
   def __init__(self,output_folder, file_name):
      if not os.path.exists(output_folder):
         os.makedirs(output_folder)

      self.file_path = os.path.join(output_folder, file_name)
      if not os.path.exists(self.file_path):
         with open(self.file_path, 'w') as self.log_file: 
            self.log_file.write('ExperimentID,ImageName,ImageID,AnnotationID,Annotation_left,Annotation_top,Annotation_right,Annotation_bottom\n')
         
   def append(self, ExperimentID, ImageName, ImageID, AnnotationID, Annotation_left, Annotation_top, Annotation_right, Annotation_bottom):
      with open(self.file_path, 'a') as log_file: 
         log_file.write('%d, %s, %d, %d, %d, %d, %d, %d\n' %\
         (ExperimentID, ImageName, ImageID, AnnotationID, Annotation_left, Annotation_top, Annotation_right, Annotation_bottom))


class log_features_live():
   def __init__(self,output_folder, file_name):
      if not os.path.exists(output_folder):
         os.makedirs(output_folder)

      self.file_path = os.path.join(output_folder, file_name)
      if not os.path.exists(self.file_path):
         with open(self.file_path, 'w') as self.log_file: 
            self.log_file.write('ExperimentID,ImageName,ImageID,FeatureID,Feature_TypeID,Feature_left,Feature_top,Feature_right,Feature_bottom\n')
         
   def append(self, ExperimentID, ImageName, ImageID, FeatureID, Feature_TypeID, Feature_left, Feature_top, Feature_right, Feature_bottom):
      with open(self.file_path, 'a') as log_file: 
         log_file.write('%d, %s, %d, %d, %d, %d, %d, %d, %d\n' %\
         (ExperimentID, ImageName, ImageID, FeatureID, Feature_TypeID, Feature_left, Feature_top, Feature_right, Feature_bottom))


# =================================================================================
# == Import dataset
# =================================================================================
dataset_dir = "../datasets/INRIAPerson_original/Test/pos/"#"../datasets/COCO_legs/images/test/"#"../datasets/INRIAPerson/Test/pos/"#
dataset_ground_truth_dir = "../datasets/INRIAPerson_original/Test/annotations/"#"../datasets/COCO_legs/labels/test/"#"../datasets/INRIAPerson/Test/annotations/"

logs_file_name = "LogsPredictions" + str(ExperimentID)+".csv"
experiment_predictions_logs = log_predictions_live(results_dir_name,logs_file_name)

logs_file_name = "LogsAnnotations" + str(ExperimentID)+".csv"
experiment_annotations_logs = log_annotations_live(results_dir_name,logs_file_name)

logs_file_name = "LogsFeatures" + str(ExperimentID)+".csv"
experiment_features_logs = log_features_live(results_dir_name,logs_file_name)

try:
   # Loop over images
   image_summary_array = []
   image_counter = 0
   image_counter_resume = 0
   number_of_images = len(os.listdir(dataset_dir))
   # for image_name in range(1):
   for image_name in os.listdir(dataset_dir):


      image_counter += 1
      print("==================")
      print("image_name = ", image_name)
      print("===================")
      print("image_counter = "+str(image_counter)+"/"+str(number_of_images))
      
      if image_counter >= image_counter_resume:
         
         image_path = dataset_dir + image_name

         image_summary = image_summary_class(image_name, image_counter)

         # image_path = "../datasets/persons_selected/1_person/person_004.jpg"

         # load and prepare image
         input_w, input_h = 416, 416

         image, image_w, image_h = load_image_pixels(image_path, (input_h, input_w))
         height, width, channels = image[0].shape

         image_cv2 = cv2.imread(image_path)
         dsize = (input_w, input_h)
         image_cv2 = cv2.resize(image_cv2, dsize)

         height_cv2, width_cv2, channels_cv2 = image_cv2.shape

         plt.clf()
         plt.imshow(image[0])
         # plt.plot([test_x1,test_x2],[test_y1,test_y2],color="black")
         plt.grid(False)
         
         # =================================================================================
         # == Generate predictions
         # =================================================================================
         v_boxes, v_labels, v_scores, box_classes_scores = YoloPredict(image, input_w, input_h, min_class_threshold)
         annotated_image = image_cv2.copy()
         # draw boxes
         do_plot = False
         # array_of_predictions = []

         counter = 0
         for i in range(len(v_boxes)):
            if v_labels[i] == "person":
               counter += 1
               box = v_boxes[i]
               confidence = v_scores[i]
               # get coordinates
               y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
               left, top, right, bottom = x1, y1, x2, y2

               # Append it to array_of_predictions
               image_summary.array_of_predictions.append(person(counter, left, top, right, bottom))
               PredictionID = counter
               Prediction_left = left
               Prediction_top = top
               Prediction_right = right
               Prediction_bottom = bottom
               Prediction_confidence = confidence
               experiment_predictions_logs.append(ExperimentID, image_name, image_counter, PredictionID, Prediction_left, Prediction_top, Prediction_right, Prediction_bottom, Prediction_confidence)
               if do_plot:
                  # draw text and score in top left corner
                  label = "%s (%.3f)" % (v_labels[i], v_scores[i])

                  # Draw rectangle 
                  cv2.rectangle(annotated_image, (int(left), int(top)), (int(right), int(bottom)), (0, 0, 230), thickness=2)
                  font                   = cv2.FONT_HERSHEY_SIMPLEX
                  fontScale              = 1
                  fontColor              = (0,0,255)
                  thickness              = 2
                  lineType               = 2

         # =================================================================================
         # == Generate Explanation
         # =================================================================================
         object_classification = False
         if object_classification == True:
            # Generate explanation
            explained_image, explanation_found_falg = GenerateExplanation(image, input_w, input_h)
            # TODO: will need to convert here the explanation or set of generated explanations into an array of predictions in order to use in the Trustworhtiness calcuation.
         else:
            explained_image = []
            explanation_found_falg = False
         
         # =================================================================================
         # == Ground truth annotations extraction
         # =================================================================================

         # Generate prediction vs Ground Truth assessment
         counter = 0
         annotaiton_name = image_name[:-3]+"txt"
         ground_truth_path = dataset_ground_truth_dir + annotaiton_name
         # print("ground_truth_path  = ", ground_truth_path)

         with open(ground_truth_path,"r", encoding = "ISO-8859-1") as f:
            lines = f.readlines()
         print(lines)
         number_of_persons_ground_truth = lines[5][28]
         # Check no more digits are there after the first digit.

         # Get the number of grouond truth annotations.
         temp_i = 0
         while True:
            temp_i += 1
            if lines[5][28+temp_i] == ' ':
               break
            number_of_persons_ground_truth += lines[5][28+temp_i] 
         number_of_persons_ground_truth = int(number_of_persons_ground_truth)
         print("==================================")
         print("number_of_persons_ground_truth = ",number_of_persons_ground_truth)

         # Loop over the ground truth predcitions and extract boundary boxes.
         # annotated_image = image_cv2.copy()
         for temp_j in range(number_of_persons_ground_truth):
            flag_left_found   = False
            flag_top_found    = False
            flag_right_found  = False
            flag_bottom_found = False

            if temp_j < 9:
               start_column = 69
            elif temp_j >= 9:
               start_column = 70

            
            temp_i = 0
            left   = lines[17+temp_j*7][start_column]

            while True:
               temp_i += 1
               current_character = lines[17+temp_j*7][start_column+temp_i]

               if not flag_left_found and current_character != ",":
                  left += current_character
               elif not flag_left_found and current_character == ",":
                  flag_left_found = True
                  temp_i += 2
                  top = lines[17+temp_j*7][start_column+temp_i]

               elif not flag_top_found and current_character != ")":
                  top += current_character
               elif not flag_top_found and current_character == ")":
                  flag_top_found = True
                  temp_i += 5
                  right = lines[17+temp_j*7][start_column+temp_i]

               elif not flag_right_found and current_character != ",":
                  right += current_character
               elif not flag_right_found and current_character == ",":
                  flag_right_found = True
                  temp_i += 2
                  bottom = lines[17+temp_j*7][start_column+temp_i]

               elif not flag_bottom_found and current_character != ")":
                  bottom += current_character
               elif not flag_bottom_found and current_character == ")":
                  flag_bottom_found = True
                  break


            print("====================================")
            left = int(left)
            right = int(right)
            top = int(top)
            bottom = int(bottom)
            print("left = ",left)
            print("right = ",right)
            print("top = ",top)
            print("bottom = ",bottom)

            # Find original original dimensions of the input image.
            flag_original_input_w_found  = False
            flag_original_input_h_found  = False
            temp_i = 0
            original_input_w = lines[3][25]
            
            while True:
               temp_i += 1
               current_character = lines[3][25+temp_i]

               if not flag_original_input_w_found and current_character != " ":
                  original_input_w += current_character
               elif not flag_original_input_w_found and current_character == " ":
                  flag_original_input_w_found = True
                  temp_i += 3
                  original_input_h = lines[3][25+temp_i]

               elif not flag_original_input_h_found and current_character != " ":
                  original_input_h += current_character
               elif not flag_original_input_h_found and current_character == " ":
                  flag_original_input_h_found = True
                  break

            original_input_w = int(original_input_w)
            original_input_h = int(original_input_h)

            # Clalculate scaled ground truth box for scaled image.
            left   = int(left*input_w/original_input_w)
            right  = int(right*input_w/original_input_w)
            top    = int(top*input_h/original_input_h)
            bottom = int(bottom*input_h/original_input_h)

            print("====")
            print("original_input_w = ",original_input_w)
            print("original_input_h = ",original_input_h)
            print("left = ",left)
            print("right = ",right)
            print("top = ",top)
            print("bottom = ",bottom)



            # ISSUE: THE IMAGES WAS RESIZED SO THE ANNOTATION DO NOT MATCH THE PICTURE.
            # We can either input the size of each image as the original.
            # OR
            # Scale the ground truth label to fit the new size.


            # Work out the ratios based on image sizes. Look here if you need help: https://stackoverflow.com/questions/51220865/resize-bounding-box-according-to-image
            
            image_summary.array_of_ground_truth_predictions.append(person(temp_j, left, top, right, bottom))
            experiment_annotations_logs.append(ExperimentID, image_name, image_counter, temp_j, left, top, right, bottom)
            


            # if do_plot:

            #    # Draw rectangle 
            #    cv2.rectangle(annotated_image, (int(left), int(top)), (int(right), int(bottom)), (0, 230, 0), thickness=2)

            #    annotated_image_path = results_dir_name+"05/"+image_name
            #    cv2.imwrite(annotated_image_path, annotated_image)
         
   
            if do_plot:

               # Draw rectangle 
               cv2.rectangle(annotated_image, (int(left), int(top)), (int(right), int(bottom)), (0, 230, 0), thickness=2)


         # =================================================================================
         # == Ground truth VS Predictions assessment
         # =================================================================================
         # Initialise True-Positive (TP), False-Positive (FP),  False-Negative (FN)
         TP = 0
         FP = 0
         FN = 0

         IoU_threshold = 0.4#0.7

         # Loop over predictions
         for current_person_prediction in image_summary.array_of_predictions:
            
            # Extract prediction box bouandaries
            prediction_left = current_person_prediction.left
            prediction_top = current_person_prediction.top
            prediction_right = current_person_prediction.right
            prediction_bottom = current_person_prediction.bottom

            current_person_prediction.P_flag = True
            current_person_prediction.G_flag = False
            
            # Loop over ground truth annotations
            for current_person_annotation in image_summary.array_of_ground_truth_predictions:

               # Extract annotation box bouandaries
               annotation_left = current_person_annotation.left
               annotation_top = current_person_annotation.top
               annotation_right = current_person_annotation.right
               annotation_bottom = current_person_annotation.bottom
               
               # Calculate Intersection over Union (IoU)
               IoU, I_left, I_top, I_right, I_bottom =  IoU_calculator(prediction_left, prediction_top, prediction_right, prediction_bottom,
                                    annotation_left, annotation_top, annotation_right, annotation_bottom)

               print("IoU = ", IoU)
               if IoU >= IoU_threshold:
                  current_person_prediction.G_flag = True
                  TP += 1 

            # After looping over all ground truths
            if current_person_prediction.G_flag == False:
               FP += 1

         # Calcualte FN
         M_predictions = len(image_summary.array_of_predictions)
         N_annotations = len(image_summary.array_of_ground_truth_predictions)
         
         FN = N_annotations - TP

         print("ground_truth_TP = ",TP)
         print("ground_truth_FP = ",FP)
         print("ground_truth_FN = ",FN)
         image_summary.ground_turth_TP = TP
         image_summary.ground_turth_FP = FP
         image_summary.ground_turth_FN = FN
         
         
         
         
         # =================================================================================
         # == Detect Features Specifications
         # =================================================================================

         # Extract features: Face
         if Face_extraction:
            ExtractFace(image_cv2, annotated_image,image_name,results_dir_name, image_summary)

         # Extract features: Palm
         if Palm_extraction:
            ExtractPalm(image_cv2, annotated_image,image_name,results_dir_name, image_summary)
         
         # Extract features: Legs
         if Legs_extraction:
            ExtractLegs(image_cv2, annotated_image,image_name,results_dir_name, image_summary)

         counter = 0
         # print("image_summary.array_of_features = ",image_summary.array_of_features)
         for feature in image_summary.array_of_features:
            counter += 1
            feature.id = counter
            experiment_features_logs.append(ExperimentID, image_name, image_counter, counter, feature.type_id, feature.left, feature.top, feature.right, feature.bottom)

         # =================================================================================
         # == Prediction VS Features Specifications assessment:
         # == Trustworthiness Cacluation
         # =================================================================================
         CalculateTrustworthiness(image, image_cv2, image_summary)
         

         image_summary_array.append(image_summary)


         if do_plot:
            annotated_image_path = results_dir_name+"05/"+image_name
            cv2.imwrite(annotated_image_path, annotated_image)

finally:
   print("Experiment finished SUCCESSFULLY!")
   pass          