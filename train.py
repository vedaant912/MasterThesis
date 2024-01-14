from config import (
    DEVICE, NUM_CLASSES,
    NUM_EPOCHS, NUM_WORKERS,
    OUT_DIR, VISUALIZE_TRANSFORMED_IMAGES
)

from dataset_dev import (
    create_train_dataset, create_valid_dataset,
    create_train_loader, create_valid_loader
)

from utils.engine import (
     train_one_epoch, evaluate
)

from models.fasterrcnn_resnet18 import create_model_resnet18, create_model_resnet34

from custom_utils import (
    save_model,
    save_train_loss_plot,
    Averager, show_tranformed_image
)

import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

if __name__ == '__main__':

    train_dataset = create_train_dataset()
    valid_dataset = create_valid_dataset()
    train_loader = create_train_loader(train_dataset, NUM_WORKERS)
    valid_loader = create_valid_loader(valid_dataset, NUM_WORKERS)
    print(f'Number of training samples : {len(train_dataset)}')
    print(f'Number of validation samples : {len(valid_dataset)}\n')

    if VISUALIZE_TRANSFORMED_IMAGES:
        show_tranformed_image(train_loader)

    # Initialize the Averager Class
    train_loss_hist = Averager()

    # Train and validation loss lists to store loss values of all
    # iterations till end and plot graphs for all iterations.
    train_loss_list = []

    # Initialize the model and move to the computation device.
    model = create_model_resnet34(num_classes=NUM_CLASSES)

    # model_load = False
    # if model_load:
    #     print('Loading the trained model....')
    #     checkpoint = torch.load('./outputs/last_model_1.pth')
    #     model.load_state_dict(checkpoint['model_state_dict'])

    model = model.to(DEVICE)

    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    print(f'{total_trainable_params:,} training parameters.\n')

    # Get the model parameters.
    params = [p for p in model.parameters() if p.requires_grad]

    # Define the optimizer
    optimizer = torch.optim.AdamW(params, lr=0.0001, weight_decay=0.0005)

    # Learning rate will be zero as we approach 'steps' number of epochs each time.
    # If 'steps = 5', LR will slowly reduce to zero every 5 epochs.
    steps = NUM_EPOCHS + 25
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=steps,
        T_mult=1,
        verbose=True
    )

    early_stopping_patience = 10
    early_stopping_counter = 0
    best_val_metric = -1

    for epoch in range(NUM_EPOCHS):
        train_loss_hist.reset()

        _, batch_loss_list = train_one_epoch(
            model,
            optimizer,
            train_loader,
            DEVICE,
            epoch,
            train_loss_hist,
            print_freq=100,
            scheduler=scheduler
        )

        evaluator = evaluate(model, valid_loader, device=DEVICE)
        
        # Add the current epoch's batch-wise losses to the 'train_loss_list'
        train_loss_list.extend(batch_loss_list)
        
        # Save the current epoch model.
        save_model(OUT_DIR, epoch, model, optimizer)
        # Save loss plot.
        save_train_loss_plot(OUT_DIR, train_loss_list)

        items_dict = evaluator.coco_eval.items()
        for a,b in items_dict:
            stats = b.stats

        AP_50 = stats[1]

        if AP_50 > best_val_metric:
            best_val_metric = AP_50
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        # Print training/validation information
        print(f'Epoch {epoch + 1}/{NUM_EPOCHS}, Validation AP_50: {AP_50:.4f}')

        # Check if early stopping criteria are met
        if early_stopping_counter >= early_stopping_patience:
            print(f'Early stopping after {epoch + 1} epochs.')
            break
