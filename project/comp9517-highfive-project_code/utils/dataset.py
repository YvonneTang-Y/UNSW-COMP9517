"""
Title: COMP9517 23T2 Group Project

Author: High Five

Description: functions/classes for processing Penguins-vs-Turts dataset
"""
import os
import PIL
import json
import torch
import shutil
import torchvision.transforms.functional as F
import torchvision.transforms.transforms as T

def rename_img_name(img_path, new_img_path):
  files = sorted(os.listdir(img_path))

  # create a new floder for new img path
  if not os.path.exists(new_img_path):
    os.makedirs(new_img_path)

  for f in files:
    new_name = f.replace('image_id_', 'image_id_0')
    shutil.copy2(os.path.join(img_path, f), os.path.join(new_img_path, new_name))

def annotation_convert(annotation_file, img_path, output_path):
  with open(annotation_file, 'r') as ann_file:
    output_path = os.path.join(output_path, "labels")
    if os.path.exists(output_path):
      shutil.rmtree(output_path)
    os.mkdir(output_path)
    
    annotation_list = json.loads(ann_file.read())
    
    for annotation in annotation_list:
      ann_file_name = "image_id_{0:04d}.txt".format(annotation['image_id'])

      output_file = os.path.join(output_path, ann_file_name)
      ann_entry = " ".join(map(str, [annotation["category_id"]] + annotation["bbox"]))

      if os.path.isfile(output_file):
        with open(os.path.join(output_path, ann_file_name), "a") as o:
          print(ann_entry, file=o)
      else:
        with open(os.path.join(output_path, ann_file_name), "w") as o:
          print(ann_entry, file=o)

def txt_to_dict(txt_path):
  with open(txt_path, "r") as txt_file:
    ann_entries = txt_file.readlines()
    boxes = []
    labels = []

    for ann_entry in ann_entries:
      entry_list = [int(v) for v in ann_entry.replace("\n", "").split(" ")]
      boxes.append(entry_list[1:])
      labels.append(entry_list[0])
    
    return boxes, labels

def bbox_format(boxes, in_format, out_format):
  ret_boxes = []

  for bbox in boxes:
    if in_format == 'xywh' and out_format == 'xyxy':
      x1, y1, w, h = bbox
      ret_boxes.append([x1, y1, x1 + w, y1 + h])
    elif in_format == 'xyxy' and out_format == 'xywh':
      x1, y1, x2, y2 = bbox
      ret_boxes.append([x1, y1, x2-x1, y2-y1])

  return ret_boxes


'''
This piece of code is copied and modified from: 
https://medium.com/@natsunoyuki/teaching-a-model-to-become-an-expert-at-locating-cats-and-dogs-in-images-716cdbc8d48f
'''
class PenguinTurtleDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, ann_dir, transforms = None):
        """
        Inputs
            root: str
                Path to the data folder.
            transforms: Compose or list
                Torchvision image transformations.
        """
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.transforms = transforms
        self.files = sorted(os.listdir(img_dir))
        
        for i in range(len(self.files)):
            self.files[i] = self.files[i].split(".")[0]
        
        print("Build dataset for \"{}\" successfully!".format(self.img_dir))
    
    def __getitem__(self, i):
        # Load image from the hard disc.
        img = PIL.Image.open(os.path.join(self.img_dir, "image_id_{0:04d}.jpg".format(i))).convert("RGB")
        
        # Load annotation file from the hard disc.
        boxes, labels = txt_to_dict(os.path.join(self.ann_dir, "image_id_{0:04d}.txt".format(i)))
        
        boxes = bbox_format(boxes, "xywh", "xyxy")
    
        # The target is given as a dict.
        target = {
            "image_id": torch.as_tensor(i),
            "boxes": torch.as_tensor(boxes, dtype = torch.float32),
            "labels": torch.as_tensor(labels, dtype = torch.int64)
        }
        
        # Apply any transforms to the data if required.
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target
    
    def __len__(self):
        return len(self.files)

class Compose:
  """
  Composes several torchvision image transforms 
  as a sequence of transformations.
  Inputs
    transforms: list
      List of torchvision image transformations.
  Returns
    image: tensor
    target: dict
  """
  def __init__(self, transforms = []):
    self.transforms = transforms

  # __call__ sequentially performs the image transformations on
  # the input image, and returns the augmented image.
  def __call__(self, image, target):
    for t in self.transforms:
      image, target = t(image, target)
    return image, target

class ToTensor(torch.nn.Module):
  """
  Converts a PIL image into a torch tensor.
  Inputs
    image: PIL Image
    target: dict
  Returns
    image: tensor
    target: dict
  """
  def forward(self, image, target = None):
    image = F.pil_to_tensor(image)
    image = F.convert_image_dtype(image)
    return image, target

class RandomHorizontalFlip(T.RandomHorizontalFlip):
  """
  Randomly flips an image horizontally.
  Inputs
    image: tensor
    target: dict
  Returns
    image: tensor
    target: dict
  """
  def forward(self, image, target = None):
    if torch.rand(1) < self.p:
      image = F.hflip(image)
      if target is not None:
        width, _ = F.get_image_size(image)
        target["boxes"][:, [0, 2]] = width - target["boxes"][:, [2, 0]]
    return image, target

def get_transform(train):
    """
    Transforms a PIL Image into a torch tensor, and performs
    random horizontal flipping of the image if training a model.
    Inputs
        train: bool
            Flag indicating whether model training will occur.
    Returns
        compose: Compose
            Composition of image transforms.
    """
    transforms = []

    # ToTensor is applied to all images.
    transforms.append(ToTensor())

    # The following transforms are applied only to the train set.
    if train == True:
        transforms.append(RandomHorizontalFlip(0.5))
        # Other transforms can be added here later on.
    return Compose(transforms)