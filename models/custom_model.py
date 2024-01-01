import torch as th
import torch.nn as nn
from gymnasium import spaces

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import torchvision.models as models


class CustomCNN(BaseFeaturesExtractor):

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):

        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0]

        self.resnet = models.resnet18(pretrained=True)

        self.resnet.conv1 = nn.Conv2d(n_input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.resnet(
                th.as_tensor(observation_space.sample()[None]).float()
            ).view(1, -1).shape[1]

        self.resnet18 = self.resnet
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:

        resnet_output = self.resnet(observations)
        flattened_output = resnet_output.view(resnet_output.size(0), -1)

        return self.linear(flattened_output)
    

class CustomCNN_resnet34(BaseFeaturesExtractor):

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):

        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0]

        self.resnet = models.resnet34(pretrained=True)

        self.resnet.conv1 = nn.Conv2d(n_input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.resnet(
                th.as_tensor(observation_space.sample()[None]).float()
            ).view(1, -1).shape[1]

        self.resnet34 = self.resnet
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:

        resnet_output = self.resnet(observations)
        flattened_output = resnet_output.view(resnet_output.size(0), -1)

        return self.linear(flattened_output)