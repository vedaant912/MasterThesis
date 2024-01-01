import torchvision
import torch.nn as nn

from stable_baselines3 import SAC

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

def create_model(num_classes):

    # Import pretrained model from Reinforcement Learning and create the backbone using those layers    
    model = SAC.load('./models/rl_model_pedestrian_200000_steps')

    critic_model_features_extractor = model.critic.features_extractor.resnet

    conv1 = critic_model_features_extractor[0]
    bn1 = critic_model_features_extractor[1]
    resnet18_relu = critic_model_features_extractor[2]
    resnet18_max_pool = critic_model_features_extractor[3]
    layer1 = critic_model_features_extractor[4]
    layer2 = critic_model_features_extractor[5]
    layer3 = critic_model_features_extractor[6]
    layer4 = critic_model_features_extractor[7]


    backbone = nn.Sequential(
        conv1, bn1, resnet18_relu, resnet18_max_pool,
        layer1, layer2, layer3, layer4
    )

    # Need output channels from the last convolutional layers from the features of the Faster RCNN model. It is 512 for ResNets
    backbone.out_channels = 512

    # Generate Anchors using the RPN
    anchor_generator = AnchorGenerator(
        sizes = ((32, 64, 128, 256, 512),),
        aspect_ratios = ((0.5, 1.0, 2.0),)
    )

    # Feature maps to perform RoI cropping.
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names = ['0'],
        output_size = 7,
        sampling_ratio = 2
    )

    # Final Faster RCNN Model
    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler
    )

    return model