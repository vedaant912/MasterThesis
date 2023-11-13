#!/bin/bash

python train_sac.py --model-name sac_model_resnet18 --width 520 --height 520 --repeat-action 4 --start-location random --sensor rgb --episode-length 600 --fps 10
