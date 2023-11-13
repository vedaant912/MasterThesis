import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.sac import CnnPolicy
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from carla_env import CarlaEnv
import argparse
from utils import TensorboardCallback, VideoRecorderCallback, lr_schedule
from custom_model import CustomCNN

import torchvision.models as models


def main(model_name, load_model, town, fps, im_width, im_height, repeat_action, start_transform_type, sensors, 
         enable_preview, steps_per_episode, create_pedestrian_flag, seed=7, action_type='continuous'):

    env = CarlaEnv(town, fps, im_width, im_height, repeat_action, start_transform_type, sensors,
                   action_type, enable_preview, steps_per_episode,create_pedestrian_flag, playing=False)
    
    # env = DummyVecEnv([lambda: env])
    # env = VecFrameStack(env, 4)
    
    try:
        if load_model:
            model = SAC.load(
                model_name, 
                env, 
                action_noise=NormalActionNoise(mean=np.array([0.3, 0.0]), sigma=np.array([0.5, 0.1])))
            
            model.load_replay_buffer('./sac_model_resnet18/rl_model_pedestrian_50000_steps/rl_model_pedestrian_70000_steps/rl_model_pedestrian_replay_buffer_500000_steps.pkl')

        else:
            
            policy_kwargs = dict(
                features_extractor_class=CustomCNN,
                features_extractor_kwargs=dict(features_dim=1),
                share_features_extractor=False
            )
        
            model = SAC(    
                "CnnPolicy", 
                env,
                learning_rate=1e-4, #lr_schedule(1e-4, 1e-6, 2),
                buffer_size=1000,
                batch_size=8,
                verbose=2,
                seed=seed,
                ent_coef='auto',
                train_freq = 64,
                device='cuda', 
                tensorboard_log='./pedestrian_logs',
                action_noise=NormalActionNoise(mean=np.array([0.3, 0]), sigma=np.array([0.5, 0.1])),
                gamma=0.98,
                tau=0.02,
                policy_kwargs=policy_kwargs,
                )
        

        checkpoint_callback = CheckpointCallback(
            save_freq = 10_000,
            save_path='./' + model_name + '/',
            name_prefix='rl_model_pedestrian',
            save_replay_buffer=True,
            save_vecnormalize=True
        )

        #video_recorder = VideoRecorderCallback(env, render_freq=500)

        model.learn(total_timesteps=500_000,
                    log_interval=10,
                    callback=[TensorboardCallback(1), checkpoint_callback],
                    tb_log_name=model_name
                )
        
        model.save(model_name)
            
    finally:
        
        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-name', help='name of model when saving')
    parser.add_argument('--load', type=bool, help='whether to load existing model')
    parser.add_argument('--map', type=str, default='Town02', help='name of carla map')
    parser.add_argument('--fps', type=int, default=10, help='fps of carla env')
    parser.add_argument('--width', type=int, help='width of camera observations')
    parser.add_argument('--height', type=int, help='height of camera observations')
    parser.add_argument('--repeat-action', type=int, help='number of steps to repeat each action')
    parser.add_argument('--start-location', type=str, help='start location type: [random, highway] for Town04')
    parser.add_argument('--sensor', action='append', type=str, help='type of sensor (can be multiple): [rgb, semantic]')
    parser.add_argument('--preview', action='store_true', help='whether to enable preview camera')
    parser.add_argument('--episode-length', type=int, help='maximum number of steps per episode')
    parser.add_argument('--seed', type=int, default=7, help='random seed for initialization')
    parser.add_argument('--create_pedestrian', type=bool, default=False, help='whether to spawn 150 pedestrians in the carla environment')
    
    args = parser.parse_args()
    model_name = args.model_name
    load_model = args.load
    town = args.map
    fps = args.fps
    im_width = args.width
    im_height = args.height
    repeat_action = args.repeat_action
    start_transform_type = args.start_location
    sensors = args.sensor
    enable_preview = False
    steps_per_episode = args.episode_length
    seed = args.seed
    create_pedestrian_flag = args.create_pedestrian

    main(model_name, load_model, town, fps, im_width, im_height, repeat_action, start_transform_type, sensors, 
         enable_preview, steps_per_episode, create_pedestrian_flag, seed)
