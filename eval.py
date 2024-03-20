# TODO: Out-dated. Refer to train_sac.py to revise the code. 

import gym
import time
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.sac import CnnPolicy
from stable_baselines3.common.noise import NormalActionNoise
from carla_env import CarlaEnv
import argparse
import logging

def main(model_name):
    town = 'Town02'
    fps = 10
    im_width = 520
    im_height = 520
    repeat_action = 4
    start_transform_type = 'random'
    sensors = 'rgb'
    action_type = 'continuous'
    enable_preview = True
    steps_per_episode = 600

    env = CarlaEnv(town, fps, im_width, im_height, repeat_action, start_transform_type, sensors,
                   action_type, enable_preview, steps_per_episode, create_pedestrian_flag=True, playing=False)
    

    try:
        model = SAC.load('./rl_model_pedestrian_100000_steps')
        model.load_replay_buffer('./rl_model_pedestrian_replay_buffer_100000_steps.pkl')

        logging.info('Loading the environment . . .')
        obs = env.reset()

        while True:
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)
            # print(reward, info)
            if done:
                obs = env.reset()
    finally:
        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-name', help='name of model when saving')

    args = parser.parse_args()
    model_name = args.model_name

    main(model_name)