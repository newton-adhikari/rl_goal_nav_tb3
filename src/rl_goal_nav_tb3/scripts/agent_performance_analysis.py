#!/usr/bin/env python3
import rclpy
import numpy as np
import matplotlib.pyplot as plt

from rl_goal_nav_tb3.rl_goal_nav_tb3_env import RLGoalNavTB3Env

# trying to create a moving obstacle for training(Gpted)
class MovingObstacle:
    def __init__(self, x, y, vx, vy, radius=0.3):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.radius = radius
        self.bounds = [-2.0, 2.0, -2.0, 2.0]  # x_min, x_max, y_min, y_max
        
    def update(self, dt=0.05):
        # update obstacle position
        self.x += self.vx * dt
        self.y += self.vy * dt
        
        # Bounce off walls
        if self.x <= self.bounds[0] or self.x >= self.bounds[1]:
            self.vx *= -1
            self.x = np.clip(self.x, self.bounds[0], self.bounds[1])
        if self.y <= self.bounds[2] or self.y >= self.bounds[3]:
            self.vy *= -1
            self.y = np.clip(self.y, self.bounds[2], self.bounds[3])
    
    def get_position(self):
        return np.array([self.x, self.y])