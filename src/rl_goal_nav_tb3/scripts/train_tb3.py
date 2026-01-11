#!/usr/bin/env python3
import rclpy
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
import os
import gymnasium as gym
import torch
from rl_goal_nav_tb3.rl_goal_nav_tb3_env import RLGoalNavTB3Env

class ROS2EnvWrapper(gym.Wrapper):
    def __init__(self):
        rclpy.init()
        env = RLGoalNavTB3Env()
        super().__init__(env)
        self.env = env

    def step(self, action):
        rclpy.spin_once(self.env, timeout_sec=0.01)
        return self.env.step(action)
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
    
    def close(self):
        self.env.close()
        rclpy.shutdown()

# quick function to create env
def make_env():
    return ROS2EnvWrapper()


def train():
    print(f"::::called the method train:::")

    # save the training logs
    log_dir = "./logs_visual/"
    save_dir = "./models_visual/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    # using gpu only if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # create dummy vec env
    # NOTE : Parallel env caused issue
    env = DummyVecEnv([make_env])

    # define save point
    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path=save_dir,
        name_prefix='tb3_goal_nav'
    )

    # define hyperparmeters for training
    model = PPO(
        'MlpPolicy', # using MlpPolicy
        env,
        learning_rate=5e-4,
        n_steps=1024,              
        batch_size=128,            
        n_epochs=5,                
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=log_dir,
        device=device,
        policy_kwargs=dict(
            net_arch=[128, 128],  
        )
    )

    print("Started training:::::")

    try:
        model.learn(
            total_timesteps=300000,
            callback=checkpoint_callback,
            progress_bar=True
        )
        
        model.save(os.path.join(save_dir, "gola_nav_agent"))
        print(f"Training completed! Model saved to {save_dir}")

    except KeyboardInterrupt:
        print("Training interrupted:::")
        model.save(os.path.join(save_dir, "gola_nav_agent_interrupted"))

    finally:
        env.close()

if __name__ == '__main__':
    train() 