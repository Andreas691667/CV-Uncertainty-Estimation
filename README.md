# Uncertainty Estimation in Object Detection
The goal of Uncertainty Estimation in Object Detection is to quantify the confidence of 
detected objects in an image, helping to assess reliability, reduce false positives, and improve 
decision-making.

- Input: An image containing multiple objects and a deep learning-based object detector that outputs object predictions.
- Output: Bounding boxes around detected objects and class labels with uncertainty scores.

## Git submodules
This repository uses git submodules to include the code of the object detection models. To initialize the submodules, run the following command:
```bash
git submodule update --init --recursive
```
