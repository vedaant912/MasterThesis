from config import (
    DEVICE, NUM_CLASSES,
    NUM_EPOCHS, NUM_WORKERS,
    OUT_DIR, VISUALIZE_TRANSFORMED_IMAGES
)

from custom_utils import (
    save_model,
    save_train_loss_plot,
    Averager, show_tranformed_image
)

from models.fasterrcnn_resnet18 import create_model

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from dataset_trial import PedestrianDataset
import matplotlib.pyplot as plt

def imshow(img):

    img = img / 2 + 0.5
    plt.imshow(np.transpose(img, (1, 2, 0)))

def previewSomeImages(loader):

    dataiter = iter(loader)

    images, labels = next(dataiter)
    images = images.numpy()

    print(images[0].shape)

    fig = plt.figure(figsize=(25, 8))

    images_to_display = 10
    for idx in np.arange(images_to_display):
        ax = fig.add_subplot(2, int(images_to_display/2), idx+1, xticks=[], yticks=[])
        imshow(images[idx])
        ax.set_title(classes[int(labels[idx])])

def calculate_accuracy(model, data_loader):
    correct = 0
    total = 0

    with torch.no_grad():
        for data in data_loader:
            images, labels = data

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    return accuracy

if __name__ == '__main__':

    class_names = ['pedestrian']
    class_names_label = {class_name:i for i, class_name in enumerate(class_names)}

    data_transform = transforms.Compose([
        transforms.Resize((520,520)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(
            degrees=(-5,5), translate=(0, 0.1), scale=(1.0, 1.25), shear=(-10, 10)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset_dir = './dataset/'

    train_dir = dataset_dir + 'Train/'
    test_dir = dataset_dir + 'Test/'
    val_dir = dataset_dir + 'Val/Val'

    train_data = PedestrianDataset(train_dir, transform=data_transform)
    test_data = PedestrianDataset(test_dir, transform=data_transform)
    val_data = PedestrianDataset(val_dir, transform=data_transform)

    batch_size = 32
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

    classes = class_names

    model = create_model(num_classes=NUM_CLASSES)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=1e-4)

    epochs = 20
    steps = 0
    print_every = 20
    running_loss = 0

    train_losses, validation_losses = [], []

    for epoch in range(epochs):
        for inputs, lables in train_loader:

            steps += 1

            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                validation_loss = 0
                accuracy = 0
                model.eval()

                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                        logps = model.forward(inputs)

                        batch_loss = criterion(logps, labels)

                        validation_loss += batch_loss.item()

                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                model.train()

                train_losses.append(running_loss/len(train_loader))
                validation_losses.append(validation_loss/len(val_loader))

                print("Epoch: {}/{}.. ".format(epoch+1, epochs),
              "Training Loss: {:.3f}.. ".format(running_loss/len(train_loader)),
              "Validation Loss: {:.3f}.. ".format(validation_loss/len(val_loader)),
              "Validation Accuracy: {:.3f}".format(accuracy/len(val_loader)))
            
                running_loss = 0
