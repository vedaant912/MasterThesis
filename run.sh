#!/bin/bash

python train_sac.py --model-name SAC_experiment_7 --width 240 --height 240 --repeat-action 4 --start-location random --sensor rgb --episode-length 600 --fps 10