from torchvision.utils import draw_bounding_boxes
import torch

from dataset_dev import (
    create_train_dataset, create_valid_dataset,
    create_train_loader, create_valid_loader
)

from config import (
    DEVICE, NUM_CLASSES,
    NUM_EPOCHS, NUM_WORKERS,
    OUT_DIR, VISUALIZE_TRANSFORMED_IMAGES
)

import matplotlib.pyplot as plt 

classes = ['background', 'pedestrian']

# Code for importing model
model = torch.load('./outputs/last_model.pth')

##########################

model.eval()
torch.cuda.empty_cache()

test_dataset = create_valid_dataset()

img, _ = test_dataset[5]
img_int = torch.tensor(img*255, dtype=torch.uint8)

with torch.no_grad():
    prediction = model([img.to(DEVICE)])
    pred = prediction[0]

fig = plt.figure(figsize=(14, 10))

plt.imshow(draw_bounding_boxes(img_int,
                               pred['boxes'][pred['scores']>0.8],
                               [classes[i] for i in pred['labels'][pred['scores']>0.8].tolist()], width=4
                               ).permute(1, 2, 0))