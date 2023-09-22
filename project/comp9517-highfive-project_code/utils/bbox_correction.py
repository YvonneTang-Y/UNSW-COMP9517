import cv2
import numpy as np
import torch
import torchvision
from pathlib import Path
from matplotlib import pyplot as plt
from collections import Counter

def show_image(image, title=None):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.xticks([]), plt.yticks([])
    plt.title(title)
    plt.show()

def get_grabcut_image(img_path, pred_bbox, display_image=False):
    #%% Show original image
    original_image = cv2.imread(img_path)
    # boundary_rectangle = tuple((pred_bbox[0]-10, pred_bbox[1]-10, pred_bbox[2]-pred_bbox[0]+20, pred_bbox[3] - pred_bbox[1]+20))
    # boundary_rectangle = tuple((pred_bbox[0]-100, pred_bbox[1]-100, pred_bbox[2]-pred_bbox[0]+100, pred_bbox[3] - pred_bbox[1]+100))
    # boundary_rectangle = tuple(pred_bbox)
    x1, y1, x2, y2 = pred_bbox
    ratio = 0.07
    boundary_rectangle = (
        max(x1 - int(ratio*(x2-x1)), 1),
        max(y1 - int(ratio*(y2-y1)), 1),
        min(x2 + int(ratio*(x2-x1)), 639) - max(x1 - int(ratio*(x2-x1)), 1),
        min(y2 + int(ratio*(y2-y1)), 639) - max(y1 - int(ratio*(y2-y1)), 1)
    )

    #%% Binarize input image
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)

    binarized_image = cv2.adaptiveThreshold(
        gray_image,
        maxValue=1,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=9,
        C=7,
    )


    #%% Use OpenCV findContours method
    contours, hierarchy = cv2.findContours(
        binarized_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    contours_image = cv2.drawContours(original_image.copy(), contours, -1, (0, 255, 0), 1)

    #%% Set GrabCut parameters
    cv2.setRNGSeed(20)
    number_of_iterations = 5

    # Define boundary rectangle containing the foreground object
    height, width, _ = original_image.shape

    #%%
    original_image_with_boundary_rectangle = cv2.rectangle(
        original_image.copy(),
        (boundary_rectangle[0], boundary_rectangle[1]),
        (boundary_rectangle[2], boundary_rectangle[3]),
        (255, 0, 0),
        1,
    )

    #%% GrabCut initialized only with a rectangle

    # Initialize mask image
    mask = np.zeros((height, width), np.uint8)
    mask[boundary_rectangle[1]:boundary_rectangle[1] + boundary_rectangle[3], 
         boundary_rectangle[0]:boundary_rectangle[0] + boundary_rectangle[2]] = 1

    # Arrays used by the algorithm internally
    background_model = np.zeros((1, 65), np.float64)
    foreground_model = np.zeros((1, 65), np.float64)

    cv2.grabCut(
        img=original_image,
        mask=mask,
        rect=boundary_rectangle,
        bgdModel=background_model,
        fgdModel=foreground_model,
        iterCount=number_of_iterations,
        mode=cv2.GC_INIT_WITH_RECT,
    )
    grabcut_mask = np.where((mask == cv2.GC_PR_BGD) | (mask == cv2.GC_BGD), 0, 1).astype(
        "uint8"
    )
    segmented_image = original_image.copy() * grabcut_mask[:, :, np.newaxis]


    # Define a kernel for morphological operations
    kernel = np.ones((3, 3), np.uint8)  # You can adjust the kernel size based on the noise characteristics

    # Apply erosion to remove small white noise regions
    eroded_image = cv2.erode(grabcut_mask, kernel, iterations=1)

    # Apply dilation to fill small gaps in the main binary regions
    denoised_image = cv2.dilate(eroded_image, kernel, iterations=1)
    
    if display_image == True:
        show_image(original_image, "Original Image")
        # show_image(denoised_image * 255, "GrabCut Mask - Noise Removed")
        # show_image(original_image_with_boundary_rectangle, "Image with boundary rectangle")
        show_image(segmented_image, "GrabCut initialized with rectangle")
        show_image(grabcut_mask * 255, "GrabCut Mask")
        show_image(denoised_image * 255, "Denoised Mask")
    
    return denoised_image

def most_common_value(lst):
    # Create a Counter object from the list
    counter = Counter(lst)

    # Use the most_common() method to get a list of tuples containing the value and its count
    most_common = counter.most_common(1)

    # Return the value with the highest count (the first element of the first tuple)
    return most_common[0]

def optimize_bbox(grabcut_mask, pred_bbox):
    width, height = 640, 640

    # Look for the leftmost white pixels in [y, y+h]
    border_pixels = []
    for i in range(pred_bbox[1], pred_bbox[3]):
        for j in range(width):
            if grabcut_mask[i, j] == 1:
                border_pixels.append(j)
                break
    leftmost_idx = min(border_pixels)
    if most_common_value(border_pixels)[1] > 0.1 * (pred_bbox[3] - pred_bbox[1]) or (pred_bbox[0] - leftmost_idx) > 0.2 * (pred_bbox[2] - pred_bbox[0]):
        leftmost_idx = pred_bbox[0]


    border_pixels = []
    for i in range(pred_bbox[1], pred_bbox[3]):
        for j in reversed(range(width)):
            if grabcut_mask[i, j] == 1:
                border_pixels.append(j)
                break
    rightmost_idx = max(border_pixels)
    if most_common_value(border_pixels)[1] > 0.1 * (pred_bbox[3] - pred_bbox[1]) or (rightmost_idx - pred_bbox[2]) > 0.2 * (pred_bbox[2] - pred_bbox[0]):
        rightmost_idx = pred_bbox[2]

    border_pixels = []
    for j in range(pred_bbox[0], pred_bbox[2]):
        for i in range(height):
            if grabcut_mask[i, j] == 1:
                border_pixels.append(i)
                break
    topmost_idx = min(border_pixels)
    if most_common_value(border_pixels)[1] > 0.1 * (pred_bbox[2] - pred_bbox[0]) or (pred_bbox[1] - topmost_idx) > 0.2 * (pred_bbox[3] - pred_bbox[1]):
        topmost_idx = pred_bbox[1]
                
    border_pixels = []
    for j in range(pred_bbox[0], pred_bbox[2]):
        for i in reversed(range(height)):
            if grabcut_mask[i, j] == 1:
                border_pixels.append(i)
                break
    bottommost_idx = max(border_pixels)
    if most_common_value(border_pixels)[1] > 0.1 * (pred_bbox[2] - pred_bbox[0]) or (bottommost_idx - pred_bbox[3]) > 0.2 * (pred_bbox[3] - pred_bbox[1]):
        bottommost_idx = pred_bbox[3]
    
    return leftmost_idx, topmost_idx, rightmost_idx, bottommost_idx
