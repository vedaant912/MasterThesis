import numpy
from typing import Any, Dict
import gym
import pygame
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import Video
import torch as th

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose = 0):
        super().__init__(verbose)

    def _on_step(self) -> bool:

        # Log scalar value
        if self.locals['dones'][0]:
            self.logger.record("custom/total_reward", self.locals['infos'][0]['total_reward'])
            self.logger.record("custom/dist_from_start", self.locals['infos'][0]['dist_from_start'])
            self.logger.record("custom/avg_speed_kmph", self.locals['infos'][0]['avg_speed_kmph'])
            self.logger.record("custom/episode_number", self.locals['infos'][0]['episode_number'])
        return True
    

class VideoRecorderCallback(BaseCallback):
    def __init__(self, eval_env: gym.Env, render_freq: int, n_eval_episodes: int = 1, deterministic: bool=True):
        
        super().__init__()
        self._eval_env = eval_env
        self._render_freq = render_freq
        self._n_eval_episodes = n_eval_episodes
        self._deterministic = deterministic

    def _on_step(self) -> bool:
        if self.n_calls % self._render_freq == 0:
            screens = []

            def grab_screens(_locals: Dict[str, Any], _globals: Dict[str, Any]) -> None:
                screen = self._eval_env.render(mode="rgb_array")
                # PyTorch uses CxHxW vs HxWxC gym (and tensorflow) image convention
                screens.append(screen.transpose(2, 0, 1))

            evaluate_policy(
                self.model,
                self._eval_env,
                callback=grab_screens
            )

            self.logger.record(
                "trajectory/video",
                Video(th.ByteTensor([screens]), fps=40),
                exclude=("stdout","log","json","csv")
            )

        return True


