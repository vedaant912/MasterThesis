import torchvision
import torch.nn as nn

from stable_baselines3 import SAC

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

def create_model_resnet18(num_classes):

    load = False
    # Import pretrained model from Reinforcement Learning and create the backbone using those layers    
    if load:
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
    else:
        conv1 = torchvision.models.resnet18(pretrained=False).conv1
        bn1 = torchvision.models.resnet18(pretrained=False).bn1
        resnet18_relu = torchvision.models.resnet18(pretrained=False).relu
        resnet18_max_pool = torchvision.models.resnet18(pretrained=False).maxpool
        layer1 = torchvision.models.resnet18(pretrained=False).layer1
        layer2 = torchvision.models.resnet18(pretrained=False).layer2
        layer3 = torchvision.models.resnet18(pretrained=False).layer3
        layer4 = torchvision.models.resnet18(pretrained=False).layer4
    
    


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

def create_model_resnet34(num_classes):

    load = True
    # Import pretrained model from Reinforcement Learning and create the backbone using those layers    
    if load:
        model = SAC.load('./models/rl_model_pedestrian_2160000_steps')
        print('################### RL Pretrained Model created ################################')
        
        critic_model_features_extractor = model.critic.features_extractor.resnet

        conv1 = critic_model_features_extractor[0]
        bn1 = critic_model_features_extractor[1]
        resnet34_relu = critic_model_features_extractor[2]
        resnet34_max_pool = critic_model_features_extractor[3]
        layer1 = critic_model_features_extractor[4]
        layer2 = critic_model_features_extractor[5]
        layer3 = critic_model_features_extractor[6]
        layer4 = critic_model_features_extractor[7]
    else:

        print("Resnet34 model create.")

        conv1 = torchvision.models.resnet34(pretrained=False).conv1
        bn1 = torchvision.models.resnet34(pretrained=False).bn1
        resnet34_relu = torchvision.models.resnet34(pretrained=False).relu
        resnet34_max_pool = torchvision.models.resnet34(pretrained=False).maxpool
        layer1 = torchvision.models.resnet34(pretrained=False).layer1
        layer2 = torchvision.models.resnet34(pretrained=False).layer2
        layer3 = torchvision.models.resnet34(pretrained=False).layer3
        layer4 = torchvision.models.resnet34(pretrained=False).layer4
    
    backbone = nn.Sequential(
        conv1, bn1, resnet34_relu, resnet34_max_pool,
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