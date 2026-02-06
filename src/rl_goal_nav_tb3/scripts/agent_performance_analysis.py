#!/usr/bin/env python3
import rclpy
import numpy as np
import matplotlib.pyplot as plt
from visualization_msgs.msg import Marker
from gazebo_msgs.srv import SpawnEntity, DeleteEntity
from rclpy.node import Node
import time
import os
import json
from datetime import datetime
from collections import defaultdict

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

    def publish_obstacle_marker(self, obstacle):
        # published rviz marker
        # NOTE: doesn't work at this time
        # TODO Fix this
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "moving_obstacles"

        # TODO: use assigned id in place of obstacle.marker_id 
        marker.id = obstacle.marker_id 
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        marker.pose.position.x = float(obstacle.x)
        marker.pose.position.y = float(obstacle.y)
        marker.pose.position.z = 0.15
        marker.pose.orientation.w = 1.0
        
        marker.scale.x = obstacle.radius * 2
        marker.scale.y = obstacle.radius * 2
        marker.scale.z = obstacle.radius * 2
        
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 0.7
        
        marker.lifetime.sec = 1
        
        self.obstacle_pub.publish(marker)

    def republish_marker(self):
        if self.current_goal_pos is not None:
            self.publish_rviz_marker(self.current_goal_pos[0], self.current_goal_pos[1])
        self.update_obstacles()
        
    def spawn_goal_sphere(self, x, y, episode_num=0):
        if self.marker_spawned and self.current_goal_name:
            self.delete_goal_sphere()
            time.sleep(0.2)
        
        self.current_goal_name = f"goal_sphere_ep{episode_num}"
        self.current_goal_pos = [x, y]
        
        if not self.spawn_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn('Spawn service not available')
            return False
        
        goal_sdf = f"""<?xml version="1.0"?>
            <sdf version="1.6">
            <model name="{self.current_goal_name}">
                <static>true</static>
                <pose>{x} {y} 0.15 0 0 0</pose>
                <link name="link">
                <visual name="visual">
                    <geometry>
                    <sphere>
                        <radius>0.25</radius>
                    </sphere>
                    </geometry>
                    <material>
                    <ambient>0 1 0 1</ambient>
                    <diffuse>0 1 0 1</diffuse>
                    <specular>0.5 1 0.5 1</specular>
                    <emissive>0 0.8 0 1</emissive>
                    </material>
                </visual>
                </link>
            </model>
            </sdf>"""
        
        request = SpawnEntity.Request()
        request.name = self.current_goal_name
        request.xml = goal_sdf
        
        future = self.spawn_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        
        if future.result() is not None:
            self.marker_spawned = True
            print(f"🟢 GREEN GOAL SPHERE spawned at ({x:.2f}, {y:.2f})")
            return True
        else:
            self.get_logger().error(f"Failed to spawn goal sphere")
            return False
        
    def delete_goal_sphere(self):
        if not self.current_goal_name:
            return
            
        if not self.delete_client.wait_for_service(timeout_sec=1.0):
            return
            
        request = DeleteEntity.Request()
        request.name = self.current_goal_name
        
        future = self.delete_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)
        
        self.marker_spawned = False
        self.current_goal_name = None
        self.current_goal_pos = None
        
    def publish_rviz_marker(self, x, y):
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "goal"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        marker.pose.position.x = float(x)
        marker.pose.position.y = float(y)
        marker.pose.position.z = 0.15
        marker.pose.orientation.w = 1.0
        
        marker.scale.x = 0.5
        marker.scale.y = 0.5
        marker.scale.z = 0.5
        
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0
        
        marker.lifetime.sec = 1
        
        self.marker_pub.publish(marker)

# created this for comphrensive performance analysis
class PerformanceAnalyzer:
    def __init__(self, save_dir="./performance_analysis"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # TODO: Need to record training and test videos seperately with rosbag
        os.makedirs(os.path.join(save_dir, 'videos'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'attention_maps'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'trajectories'), exist_ok=True)
        
        self.episode_data = []
        self.step_data = []
        self.reward_components = defaultdict(list)
        self.attention_maps = []
        self.policy_rollouts = []
        self.comparison_data = defaultdict(list)