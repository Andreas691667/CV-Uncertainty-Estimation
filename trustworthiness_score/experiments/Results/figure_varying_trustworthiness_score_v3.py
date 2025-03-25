import os
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import pandas as pd

matplotlib.rcParams.update({'font.size': 12})

# ===========================================
# == Fill info below
# ===========================================
DATASET_NAME = 'COCO'# 'INRIA' # 'COCO' 

f_n = 19#20

# Accuracy_summary_file_name  = "results_summary_"+str(f_n)+"_1.csv"
# Consensus_summary_file_name = "results_summary_"+str(f_n)+"_2.csv"
MC_summary_file_name  = "results_summary_"+str(f_n)+"_3.csv"
TCS_summary_file_name = "results_summary_"+str(f_n)+"_4.csv"

plot_additional_consensus_as_percetage_total_TCS = False
# ===========================================
# == Read in file
# ===========================================

# Accuracy_summary_df        = pd.read_csv(Accuracy_summary_file_name)
# Consensus_summary_df       = pd.read_csv(Consensus_summary_file_name)
MC_df                = pd.read_csv(MC_summary_file_name)
TCS_df               = pd.read_csv(TCS_summary_file_name)

# TS          = np.array(Accuracy_summary_df["TCS_i_h_threshold"].values).astype(int)#np.array([1,100,200,300,400,500,600,700,1000,1200,1500,1700,2000])
# TS_F1_Score = np.array(Accuracy_summary_df["F_Measure_F_P"].values)#np.array([0.802, 0.785, 0.762, 0.742, 0.72,0.707,0.689,0.674,0.635,0.613,0.587,0.571,0.553])
# GT_F1_Score = Accuracy_summary_df["F_Measure_G_P"].values[0]

MC_noTCS_array            = np.array(MC_df["model_confidence_threshold"].values).astype(int)
TP_noTCS_array            = np.array(MC_df["TP"].values).astype(int)
FP_noTCS_array            = np.array(MC_df["FP"].values).astype(int)
FN_noTCS_array            = np.array(MC_df["FN"].values).astype(int)

fig, axs = plt.subplots(int(np.ceil(len(MC_noTCS_array)/2)), 2)
plot_counter = 0
plt_x = -1
plt_y = 0

best_Precision_TCS = []
best_Recall_TCS = []
best_F1_TCS = []
best_TCS_array = []
MC_array = []

Precision_noTCS_array = []
Recall_noTCS_array = []
F1_Score_noTCS_array = []

TP_TCS_array_bar = []
FP_TCS_array_bar = []
FN_TCS_array_bar = []

TP_noTCS_array_bar = []
FP_noTCS_array_bar = []
FN_noTCS_array_bar = []


for MC in TCS_df["model_confidence_threshold"].unique().astype(int):

	# == TCS
	# MC            = np.array(TCS_df["model_confidence_threshold"].values).astype(int)
	MC_array.append(MC)
	idx           = TCS_df[TCS_df["model_confidence_threshold"]==MC].index.values
	TCS           = np.array(TCS_df["TCS_threshold"][idx].values).astype(int)

	TP            = np.array(TCS_df["TP"][idx].values).astype(int)
	FP            = np.array(TCS_df["FP"][idx].values).astype(int)
	FN            = np.array(TCS_df["FN"][idx].values).astype(int)

	Precision     = TP/(TP+FP)
	Recall        = TP/(TP+FN)
	F1_Score      = (2 * Precision * Recall) / (Precision + Recall)

	# == Keep record of highest scoring TCS thrshold
	idx      = np.where(F1_Score == max(F1_Score))[0][0]
	# idx      = np.where(Precision == max(Precision))[0][0]

	best_Precision_TCS.append(Precision[idx])
	best_Recall_TCS.append(Recall[idx])
	best_F1_TCS.append(F1_Score[idx])
	best_TCS_array.append(TCS[idx])

	# TP_TCS_array_bar.append(TP[idx][0])
	# FP_TCS_array_bar.append(FP[idx][0])
	# FN_TCS_array_bar.append(FN[idx][0])

	# == no TCS
	
	idx      = np.where(MC_noTCS_array == MC)
	print(idx)
	TP_noTCS = TP_noTCS_array[idx] * np.ones(len(TCS))
	FP_noTCS = FP_noTCS_array[idx] * np.ones(len(TCS))
	FN_noTCS = FN_noTCS_array[idx] * np.ones(len(TCS))

	TP_noTCS_array_bar.append(int(TP_noTCS[0]))
	FP_noTCS_array_bar.append(int(FP_noTCS[0]))
	FN_noTCS_array_bar.append(int(FN_noTCS[0]))

	Precision_noTCS     = TP_noTCS/(TP_noTCS+FP_noTCS)
	Recall_noTCS        = TP_noTCS/(TP_noTCS+FN_noTCS)
	F1_Score_noTCS      = (2 * Precision_noTCS * Recall_noTCS) / (Precision_noTCS + Recall_noTCS)

	Precision_noTCS_array.append(Precision_noTCS[0])
	Recall_noTCS_array.append(Recall_noTCS[0])
	F1_Score_noTCS_array.append(F1_Score_noTCS[0])

	# Plot sub plots for varying TCS for each model confidence
	if (plot_counter % 2 == 0):
		plt_x += 1
		plt_y = 0
	else:
		plt_y = 1

	axs[plt_x, plt_y].plot(TCS, Precision,'--k', linewidth='2',label= r'$Precision$') 
	# axs[plt_x, plt_y].plot(TCS, Recall, '-.k', linewidth='2',label= r'$Recall$') 
	# axs[plt_x, plt_y].plot(TCS, F1_Score, '-k', linewidth='2',label= r'$F1_Score$') 

	axs[plt_x, plt_y].plot(TCS, Precision_noTCS,'--r', linewidth='2',label= r'$Precision$') 
	# axs[plt_x, plt_y].plot(TCS, Recall_noTCS, '-.r', linewidth='2',label= r'$Recall$') 
	# axs[plt_x, plt_y].plot(TCS, F1_Score_noTCS, '-r', linewidth='2',label= r'$F1_Score$') 

	axs[plt_x, plt_y].set_title('MC = '+str(MC))
	axs[plt_x, plt_y].set_ylim([0, 1])
	plot_counter += 1

	


for ax in axs.flat:
    ax.set(xlabel='TCS threshold', ylabel='Precision')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()

plt.grid(False)
# ax.legend()
# fig.savefig(DATASET_NAME+'_exp_'+str(f_n)+'_TCS_Model_Confidence.pdf')
# fig.savefig("plots/inestigate_sus_3/"+DATASET_NAME+'_exp_'+str(f_n)+'_TCS_Model_Confidence'+'.pdf')

# == Plot summary for model confidnece with TCS and without TCS.
fig = plt.figure()
ax   = fig.add_axes([0.15,0.15,0.8,0.8])

# Plt Precision
x = MC_array
y = best_Precision_TCS
plt.plot(x, y, '-k', linewidth='1.5',label= 'TS')#'TS - Precision')#r'$Precison TCS$') 

# Plt Precision
x = MC_array
y = Precision_noTCS_array
plt.plot(x, y, '--k', linewidth='1.5',label= 'MC')#'MC - Precision')#r'$Precison MC$') 

# Plt Recall
# x = MC_array
# y = best_Recall_TCS
# plt.plot(x, y, '.-k', linewidth='2',label= r'$Recal TCS$') 
plt.ylim([0, 1.01])
plt.xlim([-1, 101])
plt.grid(False)
ax.legend()
plt.xlabel("Model Confidence Threshold")
plt.ylabel("Precision")
fig.savefig("plots/inestigate_sus_3/"+DATASET_NAME+'_exp_'+str(f_n)+'_Precision'+'.pdf')

fig = plt.figure()
ax   = fig.add_axes([0.15,0.15,0.8,0.8])

# Plt F1_Score
x = MC_array
y = best_F1_TCS
plt.plot(x, y, '-k', linewidth='1.5',label= 'TS') 

# # Plt Recall
# x = MC_array
# y = Recall_noTCS_array
# # plt.plot(x, y, '.-r', linewidth='2',label= r'$Recal No TCS$') 

# Plt F1_Score
x = MC_array
y = F1_Score_noTCS_array
plt.plot(x, y, '--k', linewidth='1.5',label= 'MC') 
plt.ylim([0, 1.01])
plt.xlim([-1, 101])
plt.grid(False)
ax.legend()
plt.xlabel("Model Confidence Threshold")
plt.ylabel("F1 Score")

display_TCS_flag = True
if display_TCS_flag == True:
	for i, v in enumerate(best_TCS_array):
		ax.text(x[i], best_F1_TCS[i]+0.1, "%d" %v, ha="center")

fig.savefig("plots/inestigate_sus_3/"+DATASET_NAME+'_exp_'+str(f_n)+'_F1_Score'+'.pdf')



# # == Plot PR curve.
# fig1 = plt.figure()
# ax1   = fig1.add_axes([0.15,0.15,0.8,0.8])


# # Plt TCS
# x = best_Recall_TCS
# y = best_Precision_TCS
# plt.plot(x, y, '--r', linewidth='2',label= r'$TCS$') 

# # Plt no TCS
# x = Recall_noTCS_array
# y = Precision_noTCS_array
# plt.plot(x, y, '.-r', linewidth='2',label= r'$No TCS$') 

# plt.xlabel("Recall")
# plt.ylabel("Precision")
# plt.grid(False)
# ax1.legend()
# fig1.savefig(DATASET_NAME+'_exp_'+str(f_n)+'_PR_curve.pdf')
