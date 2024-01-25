# Reinforcement Learning based Pretraining for Autonomous Bus Operation

## Overview

The object detection algorithm have improved a lot in the span of last two decades, majorly in last decade. The challenges faced by the early detectors that utilized handcrafted features were that their performance was subpar and have lacked the precision to be used in critical tasks as it increased the complexity. In the last decade, this has been improved by the introduction of Convolutional Neural Networks (CNN) in the field of Object Detection. The basic approach of CNN for detecting objects is by learning the features present in the image. The CNN architecture is a deep neural network architecture, where different features of the images are learnt at different hidden layers. In addition to learning the features from the images, this master thesis aims to inculcate the behavioral knowledge into the object detection model and evaluate if the model performs better. In order to achieve this, the behavioral knowledge is acquired using Reinforcement Learning (RL) algorithm. In this project, Deep Reinforcement Learning is used, which comprises of neural networks that learn to take decisions that are favorable in those complex environments. Here the goal of the RL algorithm is to reach a goal state and maximize the rewards. To obtain the behavioral knowledge, the aim is to learn to drive a vehicle autonomously in presence of pedestrian in a simulated environment and then use this trained network as a backbone in an object detection task. The RL agent is trained to drive in this conditions on the basis of rewards and penalties received while trying to navigate through the environment. It learns by trial and error method. Different parameters such as distance from midline, distance to nearest pedestrian, speed, and angle with the road are taken into consideration for designing the reward function. After the training of the RL agent, it learns to navigate through the environment in presence of the pedestrians and this network learns the behavior of the agent as well as the pedestrian around it. The aim is to transfer this knowledge to object detection backbone and train the model for detecting pedestrians in same environment.

## Requirements

This work requires different components which are listed below:

1. CARLA Environment Setup
2. Stable-baselines3 library for Reinforcement Learning Algorithm (SAC)
3. OpenAI Gym to create Custom Environment (CarlaEnv)

[CARLA](http://carla.org/) (0.9.13 version) is used in this project. Town02 is used in this project as the urban environment.

## Usage

Provide instructions on how to use the code in your repository, including any necessary steps for setting up the environment and running the code.

## Data

Describe the data that you used in your study, including any preprocessing steps that were performed. If your data is too large to include in the repository, provide a link to where it can be downloaded.

## Results

Summarize the main findings and results of your study.

## Author

Vedaant Joshi

joshi@rhrk.uni-kl.de
