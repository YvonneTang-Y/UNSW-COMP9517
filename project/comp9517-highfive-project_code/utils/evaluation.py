"""
Title: COMP9517 23T2 Group Project

Author: High Five

Description: functions for evaluate the performance
"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def decode_prediction(prediction, 
                      score_threshold = 0.5, 
                      nms_iou_threshold = 0.2):
  """
  Inputs
    prediction: dict
    score_threshold: float
    nms_iou_threshold: float
  Returns
    prediction: tuple
  """
  boxes = prediction["boxes"]
  scores = prediction["scores"]
  labels = prediction["labels"]
    
  max_idx = np.argmax(scores.cpu())

  
  return (boxes[max_idx].cpu().numpy(), 
          labels[max_idx].item(), 
          scores[max_idx].item())

def od_matrix(pd_results):

  actual_labels, predicted_labels = pd_results['truth_label'].tolist(), pd_results['pred_label'].tolist()
  accuracy = accuracy_score(actual_labels, predicted_labels)
  precision = precision_score(actual_labels, predicted_labels)
  recall = recall_score(actual_labels, predicted_labels)
  f1 = f1_score(actual_labels, predicted_labels)

  ious = pd_results['iou'].tolist()
  iou_avg = sum(ious) / len(ious)
  iou_std = np.std(ious)

  dist = pd_results['center_dist'].tolist()
  dist_avg = sum(dist) / len(dist)
  dist_std = np.std(dist)

  return {
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1-Score": f1,
    "Mean of IoU": iou_avg,
    "Standard Deviation of IoU": iou_std,
    "Mean of Distance": dist_avg,
    "Standard Deviation of Distance": dist_std
  }
