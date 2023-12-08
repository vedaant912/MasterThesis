import numpy as np
import pandas as pd

import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from xml.etree import ElementTree

class PedestrianDataset(Dataset):

    def __init__(self, root_dir, transform=ToTensor()):
        self.root_dir = root_dir
        self.transform = transform

        self.annotation_dir = os.path.join(root_dir,'Annotations')
        self.images_dir = os.path.join(root_dir, 'Images')

        self.annotation_files = sorted(os.listdir(self.annotation_dir))
        self.image_files = sorted(os.listdir(self.image_dir))

    def __len__(self):
        return len(self.annotation_files)
    
    def __getitem__(self, idx):
        annotation_file = self.annotation_files[idx]
        image_file = self.image_files[idx]

        annotation_path = os.path.join(self.annotation_dir, annotation_file)
        image_path = os.path.join(self.images_dir, image_file)

        # Load Image
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # Load label from annotation

        label = None
        image = None
        bbox = None



        return (image, label, bbox)