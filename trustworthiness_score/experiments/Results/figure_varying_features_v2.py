# ===========================================
# == Read in file
# ===========================================
import os
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import pandas as pd

matplotlib.rcParams.update({'font.size': 12})

# ===========================================
# == Fill info below
# ===========================================
DATASET_NAME = 'INRIA' # 'COCO' 

f_n = 20#19#20

# Accuracy_summary_file_name  = "results_summary_"+str(f_n)+"_1.csv"
# Consensus_summary_file_name = "results_summary_"+str(f_n)+"_2.csv"
MC_summary_file_name  = "results_summary_"+str(f_n)+"_3.csv"
TCS_face_summary_file_name = "results_summary_"+str(f_n)+"_4_face.csv"
TCS_face_palm_summary_file_name = "results_summary_"+str(f_n)+"_4_face_palm.csv"
TCS_face_palm_legs_summary_file_name = "results_summary_"+str(f_n)+"_4_face_palm_legs.csv"

# ===========================================
# == Read in file
# ===========================================

# Accuracy_summary_df        = pd.read_csv(Accuracy_summary_file_name)
# Consensus_summary_df       = pd.read_csv(Consensus_summary_file_name)
MC_df                    = pd.read_csv(MC_summary_file_name)
TCS_face_df              = pd.read_csv(TCS_face_summary_file_name)
TCS_face_palm_df         = pd.read_csv(TCS_face_palm_summary_file_name)
TCS_face_palm_legs_df    = pd.read_csv(TCS_face_palm_legs_summary_file_name)

# TS          = np.array(Accuracy_summary_df["TCS_i_h_threshold"].values).astype(int)#np.array([1,100,200,300,400,500,600,700,1000,1200,1500,1700,2000])
# TS_F1_Score = np.array(Accuracy_summary_df["F_Measure_F_P"].values)#np.array([0.802, 0.785, 0.762, 0.742, 0.72,0.707,0.689,0.674,0.635,0.613,0.587,0.571,0.553])
# GT_F1_Score = Accuracy_summary_df["F_Measure_G_P"].values[0]

MC_noTCS_array            = np.array(MC_df["model_confidence_threshold"].values).astype(int)
TP_noTCS_array            = np.array(MC_df["TP"].values).astype(int)
FP_noTCS_array            = np.array(MC_df["FP"].values).astype(int)
FN_noTCS_array            = np.array(MC_df["FN"].values).astype(int)

face_best_Precision_TCS  = []
face_best_Recall_TCS     = []
face_best_F1_TCS         = []
face_best_TCS_array      = []

face_palm_best_Precision_TCS  = []
face_palm_best_Recall_TCS     = []
face_palm_best_F1_TCS         = []
face_palm_best_TCS_array      = []

face_palm_legs_best_Precision_TCS  = []
face_palm_legs_best_Recall_TCS     = []
face_palm_legs_best_F1_TCS         = []
face_palm_legs_best_TCS_array      = []

MC_array            = []

Precision_noTCS_array  = []
Recall_noTCS_array     = []
F1_Score_noTCS_array   = []


for MC in TCS_face_df["model_confidence_threshold"].unique().astype(int):

	# == TCS
	# MC            = np.array(TCS_df["model_confidence_threshold"].values).astype(int)
	MC_array.append(MC)

	# ======= face
	idx           = TCS_face_df[TCS_face_df["model_confidence_threshold"]==MC].index.values
	TCS           = np.array(TCS_face_df["TCS_threshold"][idx].values).astype(int)

	TP            = np.array(TCS_face_df["TP"][idx].values).astype(int)
	FP            = np.array(TCS_face_df["FP"][idx].values).astype(int)
	FN            = np.array(TCS_face_df["FN"][idx].values).astype(int)

	Precision     = TP/(TP+FP)
	Recall        = TP/(TP+FN)
	F1_Score      = (2 * Precision * Recall) / (Precision + Recall)

	# = Keep record of highest scoring TCS thrshold
	idx      = np.where(F1_Score == max(F1_Score))[0][0]
	# idx      = np.where(Precision == max(Precision))[0][0]

	face_best_Precision_TCS.append(Precision[idx])
	face_best_Recall_TCS.append(Recall[idx])
	face_best_F1_TCS.append(F1_Score[idx])
	face_best_TCS_array.append(TCS[idx])

	# ======= face/palm
	idx           = TCS_face_palm_df[TCS_face_palm_df["model_confidence_threshold"]==MC].index.values
	TCS           = np.array(TCS_face_palm_df["TCS_threshold"][idx].values).astype(int)

	TP            = np.array(TCS_face_palm_df["TP"][idx].values).astype(int)
	FP            = np.array(TCS_face_palm_df["FP"][idx].values).astype(int)
	FN            = np.array(TCS_face_palm_df["FN"][idx].values).astype(int)

	Precision     = TP/(TP+FP)
	Recall        = TP/(TP+FN)
	F1_Score      = (2 * Precision * Recall) / (Precision + Recall)

	# = Keep record of highest scoring TCS thrshold
	idx      = np.where(F1_Score == max(F1_Score))[0][0]
	# idx      = np.where(Precision == max(Precision))[0][0]

	face_palm_best_Precision_TCS.append(Precision[idx])
	face_palm_best_Recall_TCS.append(Recall[idx])
	face_palm_best_F1_TCS.append(F1_Score[idx])
	face_palm_best_TCS_array.append(TCS[idx])

	# ======= face/palm/legs
	idx           = TCS_face_palm_legs_df[TCS_face_palm_legs_df["model_confidence_threshold"]==MC].index.values
	TCS           = np.array(TCS_face_palm_legs_df["TCS_threshold"][idx].values).astype(int)

	TP            = np.array(TCS_face_palm_legs_df["TP"][idx].values).astype(int)
	FP            = np.array(TCS_face_palm_legs_df["FP"][idx].values).astype(int)
	FN            = np.array(TCS_face_palm_legs_df["FN"][idx].values).astype(int)

	Precision     = TP/(TP+FP)
	Recall        = TP/(TP+FN)
	F1_Score      = (2 * Precision * Recall) / (Precision + Recall)

	# = Keep record of highest scoring TCS thrshold
	idx      = np.where(F1_Score == max(F1_Score))[0][0]
	# idx      = np.where(Precision == max(Precision))[0][0]

	face_palm_legs_best_Precision_TCS.append(Precision[idx])
	face_palm_legs_best_Recall_TCS.append(Recall[idx])
	face_palm_legs_best_F1_TCS.append(F1_Score[idx])
	face_palm_legs_best_TCS_array.append(TCS[idx])



	# == no TCS
	idx      = np.where(MC_noTCS_array == MC)
	print(idx)
	TP_noTCS = TP_noTCS_array[idx] * np.ones(len(TCS))
	FP_noTCS = FP_noTCS_array[idx] * np.ones(len(TCS))
	FN_noTCS = FN_noTCS_array[idx] * np.ones(len(TCS))


	Precision_noTCS     = TP_noTCS/(TP_noTCS+FP_noTCS)
	Recall_noTCS        = TP_noTCS/(TP_noTCS+FN_noTCS)
	F1_Score_noTCS      = (2 * Precision_noTCS * Recall_noTCS) / (Precision_noTCS + Recall_noTCS)

	Precision_noTCS_array.append(Precision_noTCS[0])
	Recall_noTCS_array.append(Recall_noTCS[0])
	F1_Score_noTCS_array.append(F1_Score_noTCS[0])




# == Plot summary for model confidnece with TCS and without TCS.
fig = plt.figure()
ax   = fig.add_axes([0.15,0.15,0.8,0.8])


X = MC_array
data = [face_best_F1_TCS, face_palm_best_F1_TCS, face_palm_legs_best_F1_TCS]
fig = plt.figure()
ax = fig.add_axes([0.16,0.16,0.8,0.8])
# ax = fig.add_axes([0,0,1,1])
ax.bar([x-0.75 for x in X] , data[0], color = '0', width = 0.75, label='TS - face')
ax.bar([x+0.00 for x in X], data[1], color = '0.5', width = 0.75, label='TS - face-palm')
ax.bar([x+0.75 for x in X], data[2], color = '0.85', width = 0.75, label='TS - face-palm-legs')
ax.patch.set_edgecolor('black')  
ax.patch.set_linewidth('1') 

# Plt F1_Score for MC
x = MC_array
y = F1_Score_noTCS_array
plt.plot(x, y, '--k', linewidth='1.5',label= 'MC') 
plt.ylim([0, 1.01])
plt.xlim([-1, 101])
plt.xticks(np.arange(0, 105, step=10))
plt.grid(False)
ax.legend()
plt.xlabel("Model Confidence Threshold")
plt.ylabel("F1 Score")
fig.savefig("plots/inestigate_sus_3/"+DATASET_NAME+'_exp_'+str(f_n)+'_varying_features_'+'.pdf')

