import rclpy
import gymnasium as gym
import numpy as np
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

class RLGoalNavTB3Env(gym.Env, Node):
    def __init__(self):
        super().__init__('rl_goal_nav_tb3_env::')

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

        # Publishers and Subscribers
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

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

    def scan_callback(self, msg):
        # will put logic later
        pass
        
    def odom_callback(self, msg):
        # will put logic later
        pass

    def reset(self, msg):
        # will put logic later
        pass

    def setp(self, msg):
        # will put logic later
        pass

    def stop(self, msg):
        # Clean up
        self.destroy_node()
    