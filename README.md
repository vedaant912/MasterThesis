# Reinforcement Learning based Pretraining for Autonomous Bus Operation

## Overview

The object detection algorithm have improved a lot in the span of last two decades, majorly in last decade. The challenges faced by the early detectors that utilized handcrafted features were that their performance was subpar and have lacked the precision to be used in critical tasks as it increased the complexity. In the last decade, this has been improved by the introduction of Convolutional Neural Networks (CNN) in the field of Object Detection. The basic approach of CNN for detecting objects is by learning the features present in the image. The CNN architecture is a deep neural network architecture, where different features of the images are learnt at different hidden layers. In addition to learning the features from the images, this master thesis aims to inculcate the behavioral knowledge into the object detection model and evaluate if the model performs better. In order to achieve this, the behavioral knowledge is acquired using Reinforcement Learning (RL) algorithm. In this project, Deep Reinforcement Learning is used, which comprises of neural networks that learn to take decisions that are favorable in those complex environments. Here the goal of the RL algorithm is to reach a goal state and maximize the rewards. To obtain the behavioral knowledge, the aim is to learn to drive a vehicle autonomously in presence of pedestrian in a simulated environment and then use this trained network as a backbone in an object detection task. The RL agent is trained to drive in this conditions on the basis of rewards and penalties received while trying to navigate through the environment. It learns by trial and error method. Different parameters such as distance from midline, distance to nearest pedestrian, speed, and angle with the road are taken into consideration for designing the reward function. After the training of the RL agent, it learns to navigate through the environment in presence of the pedestrians and this network learns the behavior of the agent as well as the pedestrian around it. The aim is to transfer this knowledge to object detection backbone and train the model for detecting pedestrians in same environment.

## Requirements

This work requires different components which are listed below:

1. CARLA Environment Setup
2. Stable-baselines3 library for Reinforcement Learning Algorithm (SAC)
3. OpenAI Gym to create Custom Environment (CarlaEnv)

[CARLA](http://carla.org/) (0.9.13 version) is used in this project. Which is available from [CARLA Releases](https://github.com/carla-simulator/carla/releases). Town02 is used in this project as the urban environment. 

For setting up, clone the repository and make sure to have Python version > 3.7. To create a new environment and install the dependencies execute the one following command:

```
# using pip
pip install -r requirements.txt

# using conda
conda create --name <env_name> --file requirements.txt
```

```
conda env create -f environment.yml
```


## Usage

### Training an Agent
In order to train a new agent or continue training an agent, use the following command:

```
bash run.sh
```

run.sh consists the command to start the training and all the possible arguments that can be provided for training according to the requirements. One such instance is as below:

```
#!/bin/bash

python train_sac.py --model-name ./Experiment_Results/model_1 --width 520 --height 520 --start-location random --sensor rgb --episode-length 600 --fps 10 --load False --create_pedestrian True
```

Each argument is explained in the following table:

|Argument               |Description                                                                        |
|-----------------------|-----------------------------------------------------------------------------------|
|model-name             |Model saved with name provided in this argument.                                   |
|width                  |Width of the image                                                                 |
|height                 |Height of the image                                                                |
|start-location         |To choose between 'random' and 'highway' location as spawn points.                 |
|sensor                 |To choose between 'rgb' and 'semantic' sensor that spawns on the vehicle.          |
|episode-length         |Maximum number of steps that are allowed to be taken by the vehicle in an episode. |
|fps                    |fps of Carla Env                                                                   |
|load                   |Set to `True` to continue the training from a checkpoint.                          |
|create-pedestrian      |Set to `True` to spawn pedestrians in the carla environement.                      |
|preview                |Set to `True` to get a preview of the current environment as seen by the vehicle.  |
|seed                   |Random seed for initialization.


## Data

Describe the data that you used in your study, including any preprocessing steps that were performed. If your data is too large to include in the repository, provide a link to where it can be downloaded.

## Results

Summarize the main findings and results of your study.

## Author

Vedaant Joshi

joshi@rhrk.uni-kl.de
