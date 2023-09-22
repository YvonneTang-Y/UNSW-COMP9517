"""
Title: COMP9517 23T2 Group Project

Author: High Five

Description: functions for ploting expected diagrams
"""

"""
Input:
    truth_box:  Integer List, like [x1, y1, x2, y2]
    pred_box:   Integer List, like [x1, y1, x2, y2]
    pred_label: Integer
    pred_score: Float
    
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from utils.evaluation import decode_prediction


# Convert human readable str label to int.
label_dict = {"Penguin": 1, "Turtle" : 2}
# Convert label int to human readable str.
reverse_label_dict = {1: "Penguin", 2: "Turtle"}

def plot_comparison(image, truth_box, pred_box, pred_label, pred_score):
  fig, ax = plt.subplots(figsize = [5, 5])
  ax.imshow(image)
  ax.axis("off")
  
  # Plot truth box
  rect = patches.Rectangle(truth_box[:2],
                            (truth_box[2] - truth_box[0]),
                            (truth_box[3] - truth_box[1]),
                            linewidth = 1,
                            edgecolor = "b",
                            facecolor = "none")
  ax.add_patch(rect)
  
  # Plot predicted box
  rect = patches.Rectangle(pred_box[:2],
                            (pred_box[2] - pred_box[0]),
                            (pred_box[3] - pred_box[1]),
                            linewidth = 1,
                            edgecolor = "r",
                            facecolor = "none")
  ax.add_patch(rect)
  
  # Display predicted label and score
  ax.text(pred_box[0],
          pred_box[1] - 5,
          "{} : {:.3f}".format(pred_label, pred_score),
          color = "r")
  plt.show()

def get_comparison_ax(ax, image, truth_box, pred_box, pred_label, pred_score):
  ax.imshow(image)
  
  # Plot truth box
  rect = patches.Rectangle(truth_box[:2],
                            (truth_box[2] - truth_box[0]),
                            (truth_box[3] - truth_box[1]),
                            linewidth = 1,
                            edgecolor = "b",
                            facecolor = "none")
  ax.add_patch(rect)
  
  # Plot predicted box
  rect = patches.Rectangle(pred_box[:2],
                            (pred_box[2] - pred_box[0]),
                            (pred_box[3] - pred_box[1]),
                            linewidth = 1,
                            edgecolor = "r",
                            facecolor = "none")
  ax.add_patch(rect)
  
  # Display predicted label and score
  ax.text(pred_box[0],
          pred_box[1] - 5,
          "{} : {:.3f}".format(pred_label, pred_score),
          color = "r")

def display_sample_results(img_ids, predictions, valid_dataset):
  # Output layout
  ncols = 4
  nrows = (len(img_ids) + ncols - 1) // ncols
  fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 10))

  for x in axs.ravel():
    x.axis("off")

  for i, img_idx in enumerate(img_ids):
    # Get the ground truth labels
    image, truth_ann = valid_dataset[img_idx]
    image = image.permute(1, 2, 0).cpu().numpy()
    truth_box = truth_ann['boxes'].cpu().numpy().reshape(-1)
    
    # Get the predicted labels
    pred_box, pred_label, pred_score = decode_prediction(predictions[img_idx])
    
    axs[i//ncols][i%ncols].set_title("image_id_{0:03d}.jpg".format(img_idx))
    get_comparison_ax(axs[i//ncols][i%ncols], image, truth_box, pred_box, reverse_label_dict[pred_label], pred_score)

  plt.show()