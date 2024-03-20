#!/bin/bash

python train_sac.py --model-name ./Experiment_Results/Experiment_13/rl_model_pedestrian_140000_steps/rl_model_pedestrian_180000_steps/rl_model_pedestrian_260000_steps/rl_model_pedestrian_320000_steps/rl_model_pedestrian_380000_steps --width 520 --height 520 --repeat-action 4 --start-location random --sensor rgb --episode-length 600 --fps 10 --load True --create_pedestrian True
