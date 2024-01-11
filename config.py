import torch

BATCH_SIZE = 4 # increase / decrease according to GPU memeory
RESIZE_TO = 512 # resize the image for training and transforms
NUM_EPOCHS = 23 # number of epochs to train for
NUM_WORKERS = 4

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  
# Images and labels direcotry should be relative to train.py
TRAIN_DIR_IMAGES = './input/train_images'
TRAIN_DIR_LABELS = './input/train_txts'
VALID_DIR_IMAGES = './input/valid_images'
VALID_DIR_LABELS = './input/valid_txts'

# classes: 0 index is reserved for background
CLASSES = [
    'background',
    'pedestrian'
]
# CLASSES = {9:'pedestrians'}

NUM_CLASSES = len(CLASSES)

# whether to visualize images after creating the data loaders
VISUALIZE_TRANSFORMED_IMAGES = True

# location to save model and plots
OUT_DIR = './outputs'

SAVE_PLOTS_EPOCH = 2 # save loss plots after these many epochs
SAVE_MODEL_EPOCH = 2 # save model after these many epochs