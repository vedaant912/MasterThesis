import torch
import cv2
import numpy as np
import os
import glob as glob

from xml.etree import ElementTree as et
from config import (
    CLASSES, RESIZE_TO, 
    TRAIN_DIR_IMAGES, VALID_DIR_IMAGES, 
    TRAIN_DIR_LABELS, VALID_DIR_LABELS,
    BATCH_SIZE
)
from torch.utils.data import Dataset, DataLoader
from custom_utils import collate_fn, get_train_transform, get_valid_transform

import ast
from utils.coco_parser import COCOParser

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torchvision

class CustomDataset(Dataset):

    def __init__(self, root, coco_annotations_file, coco_images_dir, transforms=None):
        
        self.root = root
        self.transforms = transforms
        self.coco = COCOParser(coco_annotations_file, coco_images_dir)

        self.ids = list(sorted(self.coco.get_imgIds()))

    def __getitem__(self, index):

        coco = self.coco
        
        img_id = self.ids[index]

        # List: get annotation id from coco
        ann_ids = coco.get_annIds(img_id)

        # Dictionary: target coco_annotation file for an image
        coco_annotation = coco.load_anns(ann_ids)
        
        # path for input image
        path = coco.load_imgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')

        num_objs = len(coco_annotation)

        boxes = []

        for i in range(num_objs):
            xmin = coco_annotation[i]['bbox'][0]
            ymin = coco_annotation[i]['bbox'][1]
            xmax = xmin + coco_annotation[i]['bbox'][2]
            ymax = ymin + coco_annotation[i]['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # Labels (In my case, I only one class: target class or background)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        # Tensorise img_id
        img_id = torch.tensor([int(img_id)])

        # Size of bbox (Rectangular)
        areas = []
        for i in range(num_objs):
            areas.append(coco_annotation[i]['area'])
        areas = torch.as_tensor(areas, dtype=torch.float32)
        # Iscrowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # Annotation is in dictionary format
        my_annotation = {}
        my_annotation["boxes"] = boxes
        my_annotation["labels"] = labels
        my_annotation["image_id"] = img_id
        my_annotation["area"] = areas
        my_annotation["iscrowd"] = iscrowd

        if self.transforms is not None:
            img = self.transforms(img)

        return img, my_annotation

    def __len__(self):
        return len(self.ids)
    
def get_transform():
    custom_transforms = []
    custom_transforms.append(torchvision.transforms.ToTensor())
    return torchvision.transforms.Compose(custom_transforms)

def create_train_dataset():
    train_data_dir = './input/all_pedestrians/train_images'
    coco_annotations_file="./input/all_pedestrians/train.json"
    coco_images_dir="./input/all_pedestrians/train_images"

    train_dataset = CustomDataset(root=train_data_dir,
                            coco_annotations_file=coco_annotations_file,
                            coco_images_dir=coco_images_dir,
                            transforms=get_transform()
                            )
    return train_dataset

def create_valid_dataset():
    valid_data_dir = './input/all_pedestrians/valid_images'
    coco_annotations_file="./input/all_pedestrians/valid.json"
    coco_images_dir="./input/all_pedestrians/valid_images"

    valid_dataset = CustomDataset(root=valid_data_dir,
                            coco_annotations_file=coco_annotations_file,
                            coco_images_dir=coco_images_dir,
                            transforms=get_transform()
                            )
    return valid_dataset

def create_test_dataset():
    test_data_dir = './input/all_pedestrians/test_images'
    coco_annotations_file="./input/all_pedestrians/test_temp.json"
    coco_images_dir="./input/all_pedestrians/test_images"

    test_dataset = CustomDataset(root=test_data_dir,
                            coco_annotations_file=coco_annotations_file,
                            coco_images_dir=coco_images_dir,
                            transforms=get_transform()
                            )
    return test_dataset

def create_train_loader(train_dataset, num_workers=0):
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    return train_loader

def create_valid_loader(valid_dataset, num_workers=0):
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    return valid_loader

def create_test_loader(test_dataset, num_workers=0):
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    return test_loader


if __name__ == '__main__':

    train_data_dir = './input/train_images'
    train_coco = './output.json'

    coco_annotations_file="./output.json"
    coco_images_dir="./input/train_images"

    # create own Dataset
    my_dataset = CustomDataset(root=train_data_dir,
                            coco_annotations_file=coco_annotations_file,
                            coco_images_dir=coco_images_dir,
                            transforms=get_transform()
                            )

    # collate_fn needs for batch
    def collate_fn(batch):
        return tuple(zip(*batch))
    
    # Batch size
    train_batch_size = 1

    # own DataLoader
    data_loader = torch.utils.data.DataLoader(my_dataset,
                                            batch_size=train_batch_size,
                                            shuffle=True,
                                            num_workers=4,
                                            collate_fn=collate_fn)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # DataLoader is iterable over Dataset
    for imgs, annotations in data_loader:
        imgs = list(img.to(device) for img in imgs)
        annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
        print(annotations)