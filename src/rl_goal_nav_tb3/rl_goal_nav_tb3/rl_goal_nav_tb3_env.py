import rclpy
import gymnasium as gym
import numpy as np
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from visualization_msgs.msg import Marker
from gazebo_msgs.srv import SpawnEntity, DeleteEntity

import time
import math

class RLGoalNavTB3Env(gym.Env, Node):
    def __init__(self):
        super().__init__('rl_goal_nav_tb3_env')

        # Environment parameters
        self.max_linear_vel = 0.22
        self.max_angular_vel = 2.0
        self.goal_threshold = 0.3
        self.collision_threshold = 0.2
        self.max_steps = 300

        # State variables
        self.scan_data = None
        self.position = None
        self.orientation = None
        self.goal_position = np.array([2.0, 2.0]) #initial goal position
        self.previous_distance = None
        self.step_count = 0
        self.goal_marker_spawned = False
        
        # Publishers and Subscribers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        
        # RViz marker publisher (for visualization)
        self.marker_pub = self.create_publisher(Marker, '/goal_marker', 10)
        
        # Gazebo model services (for 3D goal sphere)
        self.spawn_entity_client = self.create_client(SpawnEntity, '/spawn_entity')
        self.delete_entity_client = self.create_client(DeleteEntity, '/delete_entity')
        
        # Service clients for reset
        self.reset_simulation_client = self.create_client(Empty, '/reset_simulation')
        
        # observation_space
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0] * 24 + [-1.0, -1.0, -np.pi]),
            high=np.array([3.5] * 24 + [1.0, 1.0, np.pi]),
            dtype=np.float32
        )
        
        # action_space
        self.action_space = gym.spaces.Box(
            low=np.array([0.0, -self.max_angular_vel]),
            high=np.array([self.max_linear_vel, self.max_angular_vel]),
            dtype=np.float32
        )
        
        print("TurtleBot3 Environment initialized successfully!!!")
        
    def spawn_goal_marker(self):
        # Wait for service
        if not self.spawn_entity_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn('Spawn service not available')
            return
        
        # Delete old marker first
        if self.goal_marker_spawned:
            self.delete_goal_marker()
        
        # Create SDF for a green sphere
        goal_sdf = f"""<?xml version="1.0"?>
            <sdf version="1.6">
            <model name="goal_marker">
                <static>true</static>
                <pose>{self.goal_position[0]} {self.goal_position[1]} 0.1 0 0 0</pose>
                <link name="link">
                <visual name="visual">
                    <geometry>
                    <sphere>
                        <radius>0.15</radius>
                    </sphere>
                    </geometry>
                    <material>
                    <ambient>0 1 0 0.8</ambient>
                    <diffuse>0 1 0 0.8</diffuse>
                    </material>
                </visual>
                </link>
            </model>
            </sdf>"""
        
        # Spawn the marker
        request = SpawnEntity.Request()
        request.name = "goal_marker"
        request.xml = goal_sdf
        
        future = self.spawn_entity_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)
        
        self.goal_marker_spawned = True
        
    def delete_goal_marker(self):
        if not self.delete_entity_client.wait_for_service(timeout_sec=1.0):
            return
            
        request = DeleteEntity.Request()
        request.name = "goal_marker"
        
        future = self.delete_entity_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)
        
        self.goal_marker_spawned = False
        
    def publish_rviz_marker(self):
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "goal"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        # Position
        marker.pose.position.x = float(self.goal_position[0])
        marker.pose.position.y = float(self.goal_position[1])
        marker.pose.position.z = 0.15
        
        # Orientation
        marker.pose.orientation.w = 1.0
        
        # Scale (size)
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.3
        
        # Color (bright green with transparency)
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.8
        
        # Lifetime
        marker.lifetime.sec = 0  # 0 = forever
        
        self.marker_pub.publish(marker)
        
    def scan_callback(self, msg):
        scan_range = []
        for i in range(24): # divide lidar scans into 24 sector [making it simple]
            start_idx = i * 15
            end_idx = (i + 1) * 15
            sector_data = msg.ranges[start_idx:end_idx]
            sector_data = [x if not math.isinf(x) and not math.isnan(x) else 3.5 for x in sector_data]
            scan_range.append(min(sector_data)) # nearest obstacle
        self.scan_data = np.array(scan_range, dtype=np.float32)
        
    def odom_callback(self, msg):
        self.position = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y
        ], dtype=np.float32)

        # get orientation
        orientation_q = msg.pose.pose.orientation

        # Euler conversion
        siny_cosp = 2 * (orientation_q.w * orientation_q.z + orientation_q.x * orientation_q.y)
        cosy_cosp = 1 - 2 * (orientation_q.y * orientation_q.y + orientation_q.z * orientation_q.z)
        self.orientation = math.atan2(siny_cosp, cosy_cosp)
        
    def get_state(self):
        timeout = 0
        # wait for valid data
        # prevent invalid observations: (NaNs → network explodes), (None → crashes training), (Stale data → inconsistent learning)
        while (self.scan_data is None or self.position is None) and timeout < 20:
            rclpy.spin_once(self, timeout_sec=0.05)
            timeout += 1
            
        dx = self.goal_position[0] - self.position[0]
        dy = self.goal_position[1] - self.position[1]
        
        goal_distance = math.sqrt(dx*dx + dy*dy)
        goal_angle = math.atan2(dy, dx)
        relative_angle = math.atan2(
            math.sin(goal_angle - self.orientation),
            math.cos(goal_angle - self.orientation)
        )
        
        state = np.concatenate([
            self.scan_data,
            [dx * 0.2, dy * 0.2, relative_angle] # unscaled goal distance (to balances perception vs navigation)
        ]).astype(np.float32)
        
        return state, goal_distance
        
    def reset(self, seed=None, options=None):
        # for Gymnasium API
        #Resets Gym’s internal RNG
        super().reset(seed=seed)
        
        # Stop robot
        cmd = Twist()
        self.cmd_vel_pub.publish(cmd)
        time.sleep(0.1)
        
        # Reset simulation
        if self.reset_simulation_client.wait_for_service(timeout_sec=1.0):
            request = Empty.Request()
            future = self.reset_simulation_client.call_async(request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        
        time.sleep(0.3) # don't remove this, beause without this intitial ovservation gets NaN after reset
        
        # Generate new goal
        #TODO Avoid spawning goals inside obstacles
        #TODO Avoid goals too close to robot

        self.step_count = 0
        self.goal_position = np.array([
            np.random.uniform(-2.5, 2.5),
            np.random.uniform(-2.5, 2.5)
        ], dtype=np.float32)
        
        # Spawn visual marker in Gazebo
        self.spawn_goal_marker()
        
        # Also publish RViz marker
        self.publish_rviz_marker()
        
        state, goal_distance = self.get_state()
        self.previous_distance = goal_distance
        
        print(f"New goal at: ({self.goal_position[0]:.2f}, {self.goal_position[1]:.2f})")
        
        return state, {}
        
    def step(self, action):
        # Publish action
        cmd = Twist()
        cmd.linear.x = float(action[0])
        cmd.angular.z = float(action[1])
        self.cmd_vel_pub.publish(cmd)
        
        # Update markers every 10 steps
        # (only for visulaization very expensive on each step)
        # remove during training
        if self.step_count % 10 == 0:
            self.publish_rviz_marker()
        
        # Wait for environment
        time.sleep(0.02) #50 Hz control loop
        rclpy.spin_once(self, timeout_sec=0.005)
        
        # Get state
        state, goal_distance = self.get_state()
        self.step_count += 1

        # set reward and comleted
        reward = 0.0
        done = False
        truncated = False

        # give progress reward
        distance_diff = self.previous_distance - goal_distance
        reward += distance_diff * 50.0
        self.previous_distance = goal_distance
        
        # given Movement reward
        reward += 2.0 if action[0] > 0.05 else -1.0
        
        # given small rotation penalty
        reward -= abs(action[1]) * 0.1
        
        # given Distance-based bonus (optimized v2)
        if goal_distance < 0.5:
            reward += 10.0
        elif goal_distance < 1.0:
            reward += 5.0
        elif goal_distance < 2.0:
            reward += 2.0
            
        # Goal reached given BIG REWARD(necessary for convergence)
        if goal_distance < self.goal_threshold:
            reward += 500.0
            done = True
            print(f"Goal reached! Distance: {goal_distance:.3f}, Steps: {self.step_count}")
            
        # given collision penalty
        min_dist = np.min(self.scan_data)
        if min_dist < self.collision_threshold:
            reward -= 100.0
            done = True
            print(f"Collision! Steps: {self.step_count}")
            
        # given small time penalty (doesn't make much difference)
        # initially comemnted
        reward -= 0.01
        
        # given necessary timeout penalty
        if self.step_count >= self.max_steps:
            truncated = True
            reward -= 50.0
            
        info = {
            'goal_distance': float(goal_distance),
            'min_obstacle_distance': float(min_dist), # for better understanding of convergence
            'success': goal_distance < self.goal_threshold
        }
        
        return state, reward, done, truncated, info

    def close(self):
        # reset tb3
        cmd = Twist()
        self.cmd_vel_pub.publish(cmd)

        # Clean up
        self.destroy_node()
    