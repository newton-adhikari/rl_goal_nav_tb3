import rclpy
import gymnasium as gym
import numpy as np
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
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

        # reseting simulation for each episode
        # gazebo has to open for this to work
        self.reset_simulation_client = self.create_client(Empty, '/reset_simulation')

        # action_space
        self.action_space = gym.spaces.Box(
            low=np.array([0.0, -self.max_angular_vel]),
            high=np.array([self.max_linear_vel, self.max_angular_vel]),
            dtype=np.float32
        )

        print("TurtleBot3 Environment initialized successfully!!!")

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
        self.step_count = 0
        self.goal_position = np.array([
            np.random.uniform(-2.5, 2.5),
            np.random.uniform(-2.5, 2.5)
        ], dtype=np.float32)


        # get current state and update previous distance to the goal
        state, goal_distance = self.get_state()
        self.previous_distance = goal_distance
                
        return state, {}

    def step(self, action):
        # Publish action
        cmd = Twist()
        cmd.linear.x = float(action[0])
        cmd.angular.z = float(action[1])
        self.cmd_vel_pub.publish(cmd)
        
        # Get state
        state, goal_distance = self.get_state()
        self.step_count += 1

        # set reward and comleted
        reward = 0.0
        done = False
        truncated = False


        # give progress reward
        distance_diff = self.previous_distance - goal_distance
        reward += distance_diff * 20.0
        self.previous_distance = goal_distance


        # reward for reaching near goal
        if goal_distance < self.goal_threshold:
            reward += 500.0
            done = True
            print(f"Goal reached! Distance: {goal_distance:.3f}, Steps: {self.step_count}")

        info = {
            'goal_distance': float(goal_distance),
            'success': goal_distance < self.goal_threshold
        }

        return state, reward, done, truncated, info

    def close(self):
        # reset tb3
        cmd = Twist()
        self.cmd_vel_pub.publish(cmd)

        # Clean up
        self.destroy_node()
    