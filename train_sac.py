import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.sac import CnnPolicy
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback
from carla_env import CarlaEnv
import argparse
from utils import TensorboardCallback, VideoRecorderCallback

def main(model_name, load_model, town, fps, im_width, im_height, repeat_action, start_transform_type, sensors, 
         enable_preview, steps_per_episode, seed=7, action_type='continuous'):

    env = CarlaEnv(town, fps, im_width, im_height, repeat_action, start_transform_type, sensors,
                   action_type, enable_preview, steps_per_episode, playing=False)
    
    try:
        if load_model:
            model = SAC.load(
                model_name, 
                env, 
                action_noise=NormalActionNoise(mean=np.array([0.3, 0.0]), sigma=np.array([0.5, 0.1])))
        else:
            model = SAC(    
                CnnPolicy, 
                env,
                learning_rate=0.0003,
                buffer_size=1000,
                batch_size=8,
                verbose=2,
                seed=seed, 
                device='cuda', 
                tensorboard_log='./pedestrian_logs',
                action_noise=NormalActionNoise(mean=np.array([0.3, 0]), sigma=np.array([0.5, 0.1]))
                )
        

        checkpoint_callback = CheckpointCallback(
            save_freq = 1000,
            save_path='./pedestrian/',
            name_prefix='rl_model',
            save_replay_buffer=True,
            save_vecnormalize=True
        )

        #video_recorder = VideoRecorderCallback(env, render_freq=500)

        model.learn(total_timesteps=100_000,
                    log_interval=4,
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
    enable_preview = True
    steps_per_episode = args.episode_length
    seed = args.seed

    main(model_name, load_model, town, fps, im_width, im_height, repeat_action, start_transform_type, sensors, 
         enable_preview, steps_per_episode, seed)
