import gymnasium as gym
import torch as th
from torch import nn
import torchvision.models as models
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):

        super(CustomCombinedExtractor, self).__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0
        feature_size = 128

        for key, subspace in observation_space.spaces.items():

            if key == "image":

                n_input_channels = subspace.shape[0]
                self.resnet = models.resnet34(pretrained=False)
                self.resnet.conv1 = nn.Conv2d(n_input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
                self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

                # Compute shape by doing one forward pass
                with th.no_grad():
                    n_flatten = self.resnet(
                        th.as_tensor(observation_space['image'].sample()[None]).float()
                    ).view(1, -1).shape[1]

                self.linear = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())

                extractors[key] = nn.Sequential(self.resnet, self.linear)

                total_concat_size += feature_size #subspace.shape[1] // 4 * subspace.shape[2] // 4

            elif key == "vector":
                # Run through a simple MLP
                extractors[key] = nn.Linear(subspace.shape[0], feature_size)
                total_concat_size += feature_size

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            if key=='image':
                resnet_output = self.resnet(observations[key])
                flattened_output = resnet_output.view(resnet_output.size(0), -1)
                linear_output = self.linear(flattened_output)
                encoded_tensor_list.append(linear_output)
            else:
                encoded_tensor_list.append(extractor(observations[key]))
        
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return th.cat(encoded_tensor_list, dim=1)