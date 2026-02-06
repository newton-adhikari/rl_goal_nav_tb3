#!/usr/bin/env python3
import rclpy
import numpy as np
import matplotlib.pyplot as plt
from visualization_msgs.msg import Marker
from gazebo_msgs.srv import SpawnEntity, DeleteEntity
from rclpy.node import Node

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
    
class EnhancedVisualMarkerPublisher(Node):
    def __init__(self):
        super().__init__('enhanced_visual_marker_publisher')
        self.marker_pub = self.create_publisher(Marker, '/goal_marker', 10)
        self.obstacle_pub = self.create_publisher(Marker, '/moving_obstacles', 10)
        self.spawn_client = self.create_client(SpawnEntity, '/spawn_entity')
        self.delete_client = self.create_client(DeleteEntity, '/delete_entity')
        self.marker_spawned = False
        self.current_goal_name = None
        self.obstacle_entities = []
        
        self.marker_timer = self.create_timer(0.1, self.republish_marker)
        self.current_goal_pos = None
        self.moving_obstacles = []

        # NOTE: Started from 100 to avoid conflicts
        self.obstacle_id_counter = 100

    def add_moving_obstacle(self, obstacle):
        # actual moving obstacle added
        obstacle.marker_id = self.obstacle_id_counter
        self.obstacle_id_counter += 1
        self.moving_obstacles.append(obstacle)
        
    def update_obstacles(self):
        # update all the moving obstacles
        for obs in self.moving_obstacles:
            obs.update()
            self.publish_obstacle_marker(obs)