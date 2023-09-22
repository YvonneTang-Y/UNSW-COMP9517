import os
import cv2
import json
import random
import numpy as np
from PIL import Image

def FileInfo(floder_name):
    file_list = []
    for file in sorted(os.listdir(floder_name)):
        path = os.path.join(floder_name, file)
        # quicker to get image data
        image = Image.open(path)
        width, height = image.size

        file_info = {'name': file, 'path': path, 'width': int(width), 'height': int(height)}
        file_list.append(file_info)
    return file_list

def LabelInfo(annotation_path):
    label, box = [], []
    with open(annotation_path, 'r') as ann_file:
        annotation_list = json.loads(ann_file.read())
        for annotation in annotation_list:
            label.append(annotation["category_id"])
            box.append(annotation["bbox"])
    return label, box

# load image data
def load_image(file_dir, file_list, resize_ratio = 1):
    image_list = []
    for file in file_list:
        image = Image.open(os.path.join(file_dir, file['name']))
        image = np.array(image)[:, :, :3]
        image = cv2.resize(image, (int(file['width'] * resize_ratio), int(file['height'] * resize_ratio)), interpolation=cv2.INTER_AREA)
        image_list.append(image)
    return image_list

def load_mosaic4(index, img_list, label_list, bbox_list, path, mosaic_border = [100, 100], bbox_ratio = 0.25, image_ratio = 0.01):
    # YOLOv5 4-mosaic loader. Loads 1 image + 3 random images into a 4-image mosaic
    labels4, bbox4 = [], []
    img_size = 640
    source_indices = [i%500 for i in range(1000)]
    s = img_size
    yc, xc = (int(random.uniform(s - 4 * x, s + 4 * x)) for x in mosaic_border)  # mosaic center x, y

    indices = [index%500] + random.choices(source_indices, k=3)  # 3 additional image indices
    random.shuffle(indices)
    for i, index in enumerate(indices):
        # Load image
        img = img_list[index]
        h, w = img_size, img_size
        
        x1_bbox, y1_bbox, w_bbox, h_bbox = bbox_list[index]
        original_bbox_area = w_bbox * h_bbox
        # place img in img4
        if i == 0:  # top left
            img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            x1_new_bbox, y1_new_bbox = min(max(xc - w + x1_bbox, 0), xc), min(max(yc - h + y1_bbox, 0), yc)  # xmin, ymin (bbox)
            x2_new_bbox, y2_new_bbox = min(max(xc - w + x1_bbox + w_bbox, 0), xc), min(max(yc - h + y1_bbox + h_bbox, 0), yc) # xmax, ymax (bbox)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            x1_new_bbox, y1_new_bbox = min(max(xc + x1_bbox, xc), s * 2), min(max(yc - h + y1_bbox, 0), yc)
            x2_new_bbox, y2_new_bbox = min(max(xc + x1_bbox + w_bbox, xc), s * 2), min(max(yc - h + y1_bbox + h_bbox, 0), yc)
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            x1_new_bbox, y1_new_bbox = min(max(xc - w + x1_bbox, 0), xc), min(max(yc + y1_bbox, yc), s * 2)
            x2_new_bbox, y2_new_bbox = min(max(xc - w + x1_bbox + w_bbox, 0), xc), min(max(yc + y1_bbox + h_bbox, yc), s * 2)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            x1_new_bbox, y1_new_bbox = min(max(xc + x1_bbox, xc), s * 2), min(max(yc + y1_bbox, yc), s * 2)
            x2_new_bbox, y2_new_bbox = min(max(xc + x1_bbox + w_bbox, xc), s * 2), min(max(yc + y1_bbox + h_bbox, yc), s * 2)
        w_bbox, h_bbox = x2_new_bbox - x1_new_bbox, y2_new_bbox - y1_new_bbox
        # filter new bbox, comparing with original bbox and original image
        if w_bbox * h_bbox >= original_bbox_area * bbox_ratio and w_bbox * h_bbox >= 640 * 640 * image_ratio: 
            labels4.append(label_list[index])
            bbox4.append([int(x1_new_bbox * 0.5), int(y1_new_bbox * 0.5), int(w_bbox * 0.5), int(h_bbox * 0.5)])
        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]

    img4 = cv2.cvtColor(img4, cv2.COLOR_RGB2BGR)
    img4 = cv2.resize(img4, (s, s), interpolation=cv2.INTER_AREA)
    for bbox in bbox4:
        x, y, w, h = bbox
        cv2.rectangle(img4, (x, y), (x + w, y + h), (0, 255, 0), 2)  # plot rectangle
        
    cv2.imwrite(path, img4)
    return labels4, bbox4

def generate_mosaiced_data(input_imgs_path, input_ann_file, output_imgs_path, output_ann_file, mosaic_border = [100, 100], bbox_ratio = 0.25, image_ratio = 0.01):
  train_file_info = FileInfo(input_imgs_path)

  train_label, train_box = LabelInfo(input_ann_file)

  train_images = load_image(input_imgs_path, train_file_info)

  # load original annotations
  with open(input_ann_file, "r") as file:
    annotations = json.load(file)

  # create a new floder for mosaic
  if not os.path.exists(output_imgs_path):
    os.makedirs(output_imgs_path)

  # copy train dataset
  for i in range(0, 500):
    path = os.path.join(output_imgs_path, "image_id_{0:04d}.jpg".format(i))
    image = cv2.cvtColor(train_images[i], cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, image)

  # save new mosaiced images
  for i in range(500,1500):
    path = os.path.join(output_imgs_path, "image_id_{0:04d}.jpg".format(i))
    labels4, bbox4 = load_mosaic4(i, train_images, train_label, train_box, path, mosaic_border, bbox_ratio, image_ratio)
    for j in range(len(labels4)):  
      new_annotation = annotations[i % 500].copy()
      new_annotation['id'] = i
      new_annotation['image_id'] = i    
      new_annotation['category_id'] = labels4[j]
      new_annotation['bbox'] = bbox4[j]
      annotations.append(new_annotation)

  # create new annotations    
  new_train_annotations = json.dumps(annotations)
  with open(output_ann_file, 'w') as file:
    file.write(new_train_annotations)
