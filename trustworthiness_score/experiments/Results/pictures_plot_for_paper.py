import os
import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
import random

import cv2
from matplotlib import pyplot as plt

import matplotlib.gridspec as gridspec
# ===========================================
# == Read in file
# ===========================================

file_path_COCO  = "Experiment14/05/"
file_path_INRIA = "Experiment7/06/"


COCO_images_results = os.listdir(file_path_COCO)
INRIA_images_results = os.listdir(file_path_INRIA)

# setting values to rows and column variables
rows = 2
columns = 5
num_images =  rows*columns

plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True
gs1 = gridspec.GridSpec(rows, columns)
gs1.update(wspace=0.02, hspace=0.02)





# INRIA_to_be_used 
fig = plt.figure(figsize=(50, 20)) # create figure
for i in range(num_images):
    current_file = random.choice(INRIA_images_results)
    print(current_file)
    # INRIA_to_be_used.append(current_file)
    INRIA_images_results.remove(current_file)
    current_image =  plt.imread(file_path_INRIA+current_file)
    # Adds a subplot at the 1st position
    # fig.add_subplot(rows, columns, i)
    fig.add_subplot(gs1[i])
    # showing image
    plt.imshow(current_image)
    plt.axis('off')
# fig.savefig("INRIA_samples.png")


# COCO_to_be_used 
fig = plt.figure(figsize=(50, 20)) # create figure
for i in range(num_images):
    current_file = random.choice(COCO_images_results)
    print(current_file)
    # INRIA_to_be_used.append(current_file)
    COCO_images_results.remove(current_file)
    current_image =  plt.imread(file_path_COCO+current_file)
    # Adds a subplot at the 1st position
    # fig.add_subplot(rows, columns, i)
    fig.add_subplot(gs1[i])
    # showing image
    plt.imshow(current_image)
    plt.axis('off')
fig.savefig("COCO_samples1.png")