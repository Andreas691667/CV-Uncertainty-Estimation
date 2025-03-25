import os
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import pandas as pd

matplotlib.rcParams.update({'font.size': 12})

# ===========================================
# == Fill info below
# ===========================================
DATASET_NAME =   'INRIA' #'COCO' #'INRIA'  #'COCO' #'INRIA'  

f_n = 20#19#20#19

# Accuracy_summary_file_name  = "results_summary_"+str(f_n)+"_1.csv"
# Consensus_summary_file_name = "results_summary_"+str(f_n)+"_2.csv"
# MC_summary_file_name  = "results_summary_"+str(f_n)+"_3.csv"
# TCS_summary_file_name = "results_summary_"+str(f_n)+"_4.csv"
SUS_summary_file_name = "results_summary_"+str(f_n)+"_5.csv"

plot_additional_consensus_as_percetage_total_TCS = False
# ===========================================
# == Read in file
# ===========================================

# Accuracy_summary_df        = pd.read_csv(Accuracy_summary_file_name)
# Consensus_summary_df       = pd.read_csv(Consensus_summary_file_name)
# MC_df                = pd.read_csv(MC_summary_file_name)
# TCS_df               = pd.read_csv(TCS_summary_file_name)
SUS_df               = pd.read_csv(SUS_summary_file_name)

# # Extract variables
# MC_array                       = np.array(SUS_df["model_confidence_threshold"].values).astype(int)
# SUS_threshold_array            = np.array(SUS_df["suspicious_frame_score_threshold"].values).astype(int)

# # Extract TP, FP and FN for suspiciousness based on Features
# TP_SUS_features_array            = np.array(SUS_df["TP_suspicious_features"].values).astype(int)
# FP_SUS_features_array            = np.array(SUS_df["FP_suspicious_features"].values).astype(int)
# FN_SUS_features_array            = np.array(SUS_df["FN_suspicious_features"].values).astype(int)

# # Extract TP, FP and FN for suspiciousness based on Predictions
# TP_SUS_predictions_array            = np.array(SUS_df["TP_suspicious_predictions"].values).astype(int)
# FP_SUS_predictions_array            = np.array(SUS_df["FP_suspicious_predictions"].values).astype(int)
# FN_SUS_predictions_array            = np.array(SUS_df["FN_suspicious_predictions"].values).astype(int)

# # Extract TP, FP and FN for suspiciousness based on Features or Predictions
# TP_SUS_features_predictions_array            = np.array(SUS_df["TP_suspicious_features_and_predictions"].values).astype(int)
# FP_SUS_features_predictions_array            = np.array(SUS_df["FP_suspicious_features_and_predictions"].values).astype(int)
# FN_SUS_features_predictions_array            = np.array(SUS_df["FN_suspicious_features_and_predictions"].values).astype(int)



for based_on in ["untrustworthy_predictions"]:
	best_Precision_SUS = []
	best_Recall_SUS = []
	best_F1_SUS = []

	best_SUS_array = []

	best_Precision_SUS_MC = []
	best_Recall_SUS_MC = []
	best_F1_SUS_MC = []

	MC_array = []
	
	SUS_df_filtered = SUS_df#[SUS_df.min_MC_suspicous_threshold == min_MC_SUS_threshold]
	for MC in SUS_df_filtered["model_confidence_threshold"].unique().astype(int):

		# == SUS
		# MC            = np.array(TCS_df["model_confidence_threshold"].values).astype(int)
		MC_array.append(MC)
		idx               = SUS_df_filtered[SUS_df_filtered["model_confidence_threshold"]==MC].index.values
		SUS               = np.array(SUS_df_filtered["suspicious_frame_score_threshold"][idx].values).astype(int)

		TP_MC             = np.array(SUS_df_filtered["TP_suspicious_MC"][idx].values).astype(int)
		FP_MC             = np.array(SUS_df_filtered["FP_suspicious_MC"][idx].values).astype(int)
		FN_MC             = np.array(SUS_df_filtered["FN_suspicious_MC"][idx].values).astype(int)
		

		if based_on == "untrustworthy_predictions":
			TP            = np.array(SUS_df_filtered["TP_suspicious_predictions"][idx].values).astype(int)
			FP            = np.array(SUS_df_filtered["FP_suspicious_predictions"][idx].values).astype(int)
			FN            = np.array(SUS_df_filtered["FN_suspicious_predictions"][idx].values).astype(int)

		Precision     = np.nan_to_num(TP/(TP+FP))
		Recall        = np.nan_to_num(TP/(TP+FN))
		F1_Score      = np.nan_to_num((2 * Precision * Recall) / (Precision + Recall))

		Precision_MC     = np.nan_to_num(TP_MC/(TP_MC+FP_MC))
		Recall_MC        = np.nan_to_num(TP_MC/(TP_MC+FN_MC))
		F1_Score_MC      = np.nan_to_num((2 * Precision_MC * Recall_MC) / (Precision_MC + Recall_MC))

		# if min_MC_SUS_threshold == 70:
		print("=========")
		print(based_on)
		print("TP = ",TP)
		print(FP)
		print(FN)
		
		print(Precision)
		print(Recall)
		print(F1_Score)

		print("TP_MC = ",TP_MC)
		print("FP_MC = ",FP_MC)
		print("FN_MC = ",FN_MC)
		
		print("Precision = ",Precision_MC)
		print("Recall = ",Recall_MC)
		print("F1_Score = ",F1_Score_MC)
		


		# == Keep record of highest scoring SUS thrshold
		idx      = np.where(F1_Score == max(F1_Score))[0][0]
		# print("idx = ", idx)
		best_Precision_SUS.append(Precision[idx])
		best_Recall_SUS.append(Recall[idx])
		best_F1_SUS.append(F1_Score[idx])
		best_SUS_array.append(SUS[idx])

		idx      = np.where(F1_Score_MC == max(F1_Score_MC))[0][0]
		# print("idx = ", idx)
		best_Precision_SUS_MC.append(Precision_MC[idx])
		best_Recall_SUS_MC.append(Recall_MC[idx])
		best_F1_SUS_MC.append(F1_Score_MC[idx])



	# == Plot summary for model confidnece with TCS and without TCS.
	# if based_on == "nonallocated_features":
	# 	fig = plt.figure()
	# if based_on == "untrustworthy_predictions":
	# 	fig = plt.figure()
	# if based_on == "sum_of_both":
	# 	fig = plt.figure()
	fig = plt.figure()
	ax   = fig.add_axes([0.15,0.15,0.8,0.8])

	# Plt Precision
	x = MC_array[2:-1]
	y = best_Precision_SUS[2:-1]
	print(x)
	print(y)
	plt.plot(x, y, '-k', linewidth='1.5',label= 'SS') #'Precision SUS')

	# Plt MC Precision
	x = MC_array[2:-1]
	y = best_Precision_SUS_MC[2:-1]
	print(x)
	print(y)
	plt.plot(x, y, '--k', linewidth='1.5',label= 'MC') #'MC Precision SUS') 

	display_SUS_flag = False
	if display_SUS_flag == True:
		for i, v in enumerate(best_SUS_array[1:-1]):
			ax.text(x[i], best_Precision_SUS[i]+0.1, "%d" %v, ha="center") 

	# plt.ylim([0, 1.01])
	plt.xlim([0, 100])
	plt.grid(False)
	ax.legend()
	plt.xlabel("Model Confidence Threshold")
	plt.ylabel("Precision")
	fig.savefig("plots/inestigate_sus_4/"+DATASET_NAME+'_exp_'+str(f_n)+'_Precision_SUS_VS_MC_'+based_on+'.pdf')

	plt.figure().clear()
	plt.close()
	plt.cla()
	plt.clf()

	fig = plt.figure()
	ax   = fig.add_axes([0.15,0.15,0.8,0.8])


	# Plt F1_Score
	x = MC_array[2:-1]
	y = best_F1_SUS[2:-1]
	plt.plot(x, y, '-k', linewidth='1.5',label= 'SS')
	
	# Plt MC F1_Score
	x = MC_array[2:-1]
	y = best_F1_SUS_MC[2:-1]
	plt.plot(x, y, '--k', linewidth='1.5',label= 'MC') 

	# plt.ylim([0, 1.01])
	plt.xlim([0, 100])
	plt.grid(False)
	ax.legend()
	plt.xlabel("Model Confidence Threshold")
	plt.ylabel("F1 Score")
	fig.savefig("plots/inestigate_sus_4/"+DATASET_NAME+'_exp_'+str(f_n)+'_F1Score_SUS_VS_MC_'+based_on+'.pdf')

