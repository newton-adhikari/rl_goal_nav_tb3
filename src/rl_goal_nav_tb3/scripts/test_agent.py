#!/usr/bin/env python3
import rclpy
from stable_baselines3 import PPO, SAC
import time
import numpy as np
from rl_goal_nav_tb3.rl_goal_nav_tb3_env import RLGoalNavTB3Env

def load_model(model_path):
    try:
        return PPO.load(model_path)
    except Exception as e:
        raise Exception(f"Failed to load model: {e}")

def test_agent(model_path, num_episodes=10):
    # Initialize ROS2
    rclpy.init()
    
    # Load trained model
    print(f"Loading model from {model_path}")
    model = load_model(model_path)
    
    # Create environment
    env = RLGoalNavTB3Env()
    episode_rewards = []
    episode_steps = []
    success_count = 0
    
    try:
        for episode in range(num_episodes):
            print(f"Episode {episode + 1}/{num_episodes}")
            print('='*60)
            
            obs, _ = env.reset()
            done = False
            truncated = False
            episode_reward = 0
            steps = 0
            
            while not done and not truncated:
                # Get action from model
                action, _ = model.predict(obs, deterministic=True)
                
                # Step environment
                rclpy.spin_once(env, timeout_sec=0.01)
                obs, reward, done, truncated, info = env.step(action)
                
                episode_reward += reward
                steps += 1
                
                # progress
                if steps % 50 == 0:
                    print(f"  Step {steps}: Goal={info.get('goal_distance', 0):.3f}m, "
                          f"Min_obstacle={info.get('min_obstacle_distance', 0):.3f}m")
                
                time.sleep(0.01)
            
            # Episode summary
            goal_reached = info.get('success', info.get('goal_distance', 1.0) < env.goal_threshold)
            
            if goal_reached:
                print(f"SUCCESS! Reached goal in {steps} steps")
                success_count += 1
            else:
                print(f"FAILED after {steps} steps")
                
            print(f"Total reward: {episode_reward:.2f}")
            
            # Store metrics
            episode_rewards.append(episode_reward)
            episode_steps.append(steps)
            
            time.sleep(2)
        
        # train data
        print(f"Episodes: {num_episodes}")
        print(f"Success rate: {success_count/num_episodes*100:.1f}% ({success_count}/{num_episodes})")
        print(f"Average reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
        print(f"Average steps: {np.mean(episode_steps):.1f} ± {np.std(episode_steps):.1f}")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\nTesting interrupted by user")
        
    finally:
        env.close()
        rclpy.shutdown()

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python3 test_agent.py <model_path> [num_episodes]")
        print("  num_episodes: number of episodes (default: 10)")
        print("\nExample:")
        print("  python3 test_agent.py ./models/turtlebot3_ppo_final.zip basic 10")
        sys.exit(1)
    
    model_path = sys.argv[1]
    num_episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    
    test_agent(model_path, num_episodes)


    # python3 test_agent.py ./completed_model/turtlebot3_visual_final 50