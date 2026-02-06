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
import pandas as pd
from scipy.ndimage import gaussian_filter
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation, PillowWriter
from stable_baselines3 import PPO, SAC

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
    
    def record_episode(self, episode_num, trajectory, goal_pos, success, total_reward, steps, obstacles=None):
        # Record complete episode data
        # Convert trajectory to list for JSON serialization
        trajectory_list = [pos.tolist() if hasattr(pos, 'tolist') else list(pos) for pos in trajectory]
        
        self.episode_data.append({
            'episode': episode_num,
            'success': success,
            'total_reward': total_reward,
            'steps': steps,
            'goal_x': goal_pos[0],
            'goal_y': goal_pos[1],
            'trajectory': trajectory_list,
            'efficiency': self._calculate_efficiency(trajectory, goal_pos),
            'path_smoothness': self._calculate_smoothness(trajectory),
            'jerk': self._calculate_jerk(trajectory),
            'time_to_goal': steps * 0.05,
            'clearance': self._calculate_min_clearance(trajectory, obstacles) if obstacles else None,
            'energy_consumption': self._estimate_energy(trajectory)
        })


    def record_step(self, episode, step, state, action, reward, info):
        # Record individual step data(turned out to be useful)
        self.step_data.append({
            'episode': episode,
            'step': step,
            'distance_to_goal': info.get('goal_distance', 0),
            'min_obstacle_dist': info.get('min_obstacle_distance', 0),
            'linear_vel': action[0],
            'angular_vel': action[1],
            'reward': reward,
            'timestamp': step * 0.05
        })

    def record_reward_components(self, episode, step, components):
        # Record individual reward components
        for key, value in components.items():
            self.reward_components[key].append({
                'episode': episode,
                'step': step,
                'value': value
            })

    def record_attention(self, episode, step, scan_data, position, goal_pos):
        # Record attention data
        self.attention_maps.append({
            'episode': episode,
            'step': step,
            'scan_data': scan_data.tolist() if hasattr(scan_data, 'tolist') else list(scan_data),
            'position': position.tolist() if hasattr(position, 'tolist') else list(position),
            'goal_position': goal_pos.tolist() if hasattr(goal_pos, 'tolist') else list(goal_pos),
            'attention_weights': self._compute_attention_weights(scan_data).tolist(),
            'saliency_map': self._compute_saliency_map(scan_data, position, goal_pos).tolist()
        })

    def record_policy_rollout(self, episode, step, state, action, q_values=None):
        # this is policy decisions, very important for analysis
        self.policy_rollouts.append({
            'episode': episode,
            'step': step,
            'state': state.copy() if hasattr(state, 'copy') else state,
            'action': action.copy() if hasattr(action, 'copy') else action,
            'q_values': q_values
        })

    def visualize_attention_saliency(self, episode_idx=0, num_frames=9, save=True):
        # Visualize attention and saliency maps"""
        episode_attention = [a for a in self.attention_maps if a['episode'] == episode_idx]
        
        if not episode_attention:
            print(f"No attention data for episode {episode_idx}")
            return
        
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.3)
        
        # Sample frames across episode
        indices = np.linspace(0, len(episode_attention)-1, num_frames, dtype=int)
        
        for idx, frame_idx in enumerate(indices):
            data = episode_attention[frame_idx]
            
            # Polar plot with attention
            ax_polar = plt.subplot(gs[idx // 3, idx % 3], projection='polar')
            
            angles = np.linspace(0, 2*np.pi, 24, endpoint=False)
            # Converted back to numpy arrays
            scan = np.array(data['scan_data'])
            attention = np.array(data['attention_weights'])
            position = np.array(data['position'])
            goal_position = np.array(data['goal_position'])
            
            # Plot distance readings
            ax_polar.plot(angles, scan, 'b-', linewidth=2.5, label='LiDAR', alpha=0.7)
            ax_polar.fill_between(angles, 0, scan, color='blue', alpha=0.1)
            
            # Plot attention overlay
            attention_scaled = attention * 3  # Scale for visibility
            ax_polar.fill(angles, attention_scaled, 'r', alpha=0.4, label='Attention')
            ax_polar.plot(angles, attention_scaled, 'r-', linewidth=2)
            
            # Highlight max attention direction
            max_idx = np.argmax(attention)
            ax_polar.plot([angles[max_idx], angles[max_idx]], [0, 3.5], 
                         'g--', linewidth=2, label='Focus')
            
            ax_polar.set_ylim(0, 3.5)
            ax_polar.set_title(f'Step {data["step"]}\nDist to Goal: {np.linalg.norm(position - goal_position):.2f}m',
                             fontsize=10)
            ax_polar.legend(loc='upper right', fontsize=8)
            ax_polar.set_theta_zero_location('N')
            
        plt.suptitle(f'Attention & Saliency Analysis - Episode {episode_idx}', 
                    fontsize=16, fontweight='bold')
        
        if save:
            save_path = os.path.join(self.save_dir, 'attention_maps', f'attention_episode_{episode_idx}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f" Attention visualization saved to: {save_path}")
        
        plt.show()

        def visualize_reward_breakdown(self, save=True):
            # Detailed reward component analysis
            if not self.reward_components:
                print("No reward component data")
                return
            
            fig, axes = plt.subplots(3, 2, figsize=(16, 14))
            fig.suptitle('Reward Component Breakdown Analysis', fontsize=16, fontweight='bold')
            
            component_names = list(self.reward_components.keys())
            
            for idx, comp_name in enumerate(component_names[:6]):
                ax = axes[idx // 2, idx % 2]
                data = self.reward_components[comp_name]
                df = pd.DataFrame(data)
                
                # Episode-wise mean
                episode_means = df.groupby('episode')['value'].mean()
                episode_std = df.groupby('episode')['value'].std()
                
                # Convert to numpy arrays for compatibility
                x_vals = episode_means.index.to_numpy()
                y_vals = episode_means.values
                std_vals = episode_std.values
                
                # Plot with confidence interval
                ax.plot(x_vals, y_vals, 
                    marker='o', linewidth=2, markersize=4, label='Mean')
                ax.fill_between(x_vals,
                            y_vals - std_vals,
                            y_vals + std_vals,
                            alpha=0.3, label='Std Dev')
                
                ax.set_xlabel('Episode')
                ax.set_ylabel('Value')
                ax.set_title(f'{comp_name.replace("_", " ").title()}')
                ax.grid(True, alpha=0.3)
                ax.legend()
                
                # Add statistics
                stats_text = f'Mean: {episode_means.mean():.3f}\nStd: {episode_means.std():.3f}'
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5),
                    fontsize=8)
            
            plt.tight_layout()
            
            if save:
                save_path = os.path.join(self.save_dir, 'reward_breakdown_analysis.png')
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f" Reward breakdown saved to: {save_path}")
            
            plt.show()
    
    
    def visualize_advanced_trajectories(self, save=True):
        # Advanced trajectory visualization with heatmaps
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # 1. All Trajectories with Success/Failure
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_title('All Trajectories (Success vs Failure)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('X Position (m)')
        ax1.set_ylabel('Y Position (m)')
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        for ep_data in self.episode_data:
            trajectory = ep_data['trajectory']
            if len(trajectory) > 0:
                traj_array = np.array(trajectory)
                color = 'green' if ep_data['success'] else 'red'
                alpha = 0.4
                linewidth = 2 if ep_data['success'] else 1
                
                ax1.plot(traj_array[:, 0], traj_array[:, 1], 
                        color=color, alpha=alpha, linewidth=linewidth)
                
                # Marked goal
                ax1.add_patch(Circle((ep_data['goal_x'], ep_data['goal_y']), 
                                    0.3, color='gold', alpha=0.6, zorder=10))
        
        ax1.plot([], [], 'g-', linewidth=2, label=f"Success ({sum(e['success'] for e in self.episode_data)})")
        ax1.plot([], [], 'r-', linewidth=1, label=f"Failure ({sum(not e['success'] for e in self.episode_data)})")
        ax1.legend(loc='upper right')
        
        # 2. Trajectory Density Heatmap
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_title('Trajectory Density Heatmap', fontsize=14, fontweight='bold')
        
        # Create density map with fixed bins
        x_bins = np.linspace(-2.5, 2.5, 101)  # NOTE: Used 101 edges for 100 bins
        y_bins = np.linspace(-2.5, 2.5, 101)
        density = np.zeros((100, 100))
        
        for ep_data in self.episode_data:
            trajectory = np.array(ep_data['trajectory'])
            if len(trajectory) > 0:
                H, _, _ = np.histogram2d(trajectory[:, 0], trajectory[:, 1], 
                                         bins=[x_bins, y_bins])
                
                # NOTE: Need to fix here, H needs to be valid before adding
                if H.shape == (100, 100):
                    density += H.T
        
        # Only plot if we have data
        if density.max() > 0:
            im = ax2.imshow(density, extent=[-2.5, 2.5, -2.5, 2.5], 
                           origin='lower', cmap='hot', alpha=0.7)
            plt.colorbar(im, ax=ax2, label='Visit Frequency')
        ax2.set_xlabel('X Position (m)')
        ax2.set_ylabel('Y Position (m)')
        
        # 3. Efficiency vs Steps
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.set_title('Efficiency vs Steps Taken', fontsize=14, fontweight='bold')
        
        df = pd.DataFrame(self.episode_data)
        colors = ['green' if s else 'red' for s in df['success']]
        ax3.scatter(df['steps'].to_numpy(), df['efficiency'].to_numpy(), c=colors, alpha=0.6, s=100)
        ax3.set_xlabel('Steps Taken')
        ax3.set_ylabel('Path Efficiency')
        ax3.grid(True, alpha=0.3)
        
        # 4. Best Trajectory
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.set_title('Best Trajectory (Highest Efficiency)', fontsize=14, fontweight='bold')
        
        best_idx = df['efficiency'].idxmax()
        best_ep = self.episode_data[best_idx]
        traj = np.array(best_ep['trajectory'])
        
        ax4.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=3, label='Path')
        ax4.plot(traj[0, 0], traj[0, 1], 'go', markersize=15, label='Start')
        ax4.add_patch(Circle((best_ep['goal_x'], best_ep['goal_y']), 
                            0.3, color='gold', label='Goal'))
        ax4.set_xlabel('X Position (m)')
        ax4.set_ylabel('Y Position (m)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_aspect('equal')
        ax4.text(0.05, 0.95, f"Efficiency: {best_ep['efficiency']:.3f}\nSteps: {best_ep['steps']}", 
                transform=ax4.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 5. Time Series Analysis
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.set_title('Performance Over Episodes', fontsize=14, fontweight='bold')
        
        episodes = range(len(df))
        ax5_twin = ax5.twinx()
        
        # Converted to numpy arrays just for compatibility
        efficiency_vals = df['efficiency'].to_numpy()
        reward_vals = df['total_reward'].to_numpy()
        
        line1 = ax5.plot(episodes, efficiency_vals, 'b-', label='Efficiency', linewidth=2)
        line2 = ax5_twin.plot(episodes, reward_vals, 'r-', label='Reward', linewidth=2)
        
        ax5.set_xlabel('Episode')
        ax5.set_ylabel('Efficiency', color='b')
        ax5_twin.set_ylabel('Total Reward', color='r')
        ax5.tick_params(axis='y', labelcolor='b')
        ax5_twin.tick_params(axis='y', labelcolor='r')
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax5.legend(lines, labels, loc='upper left')
        ax5.grid(True, alpha=0.3)
        
        # 6. Success Rate Distribution
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.set_title('Performance Metrics Distribution', fontsize=14, fontweight='bold')
        
        metrics = ['Efficiency', 'Smoothness', 'Energy']
        values = [
            df['efficiency'].mean(),
            1 - df['path_smoothness'].mean(),  # NOTE: Inverted for better visualization
            1 - (df['energy_consumption'].mean() / df['energy_consumption'].max())
        ]
        
        bars = ax6.bar(metrics, values, color=['#2ecc71', '#3498db', '#e74c3c'], alpha=0.7)
        ax6.set_ylabel('Normalized Score')
        ax6.set_ylim(0, 1)
        ax6.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}', ha='center', va='bottom')
        
        plt.suptitle('Comprehensive Trajectory Analysis for Final Report', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        if save:
            save_path = os.path.join(self.save_dir, 'trajectories', 'advanced_trajectory_analysis.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f" Advanced trajectory analysis saved to: {save_path}")
        
        plt.show()
        # Generate comparison plots for final report
        df = pd.DataFrame(self.episode_data)
        
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # 1. Success Rate by Episode Batch
        ax1 = fig.add_subplot(gs[0, 0])
        batch_size = 10
        batches = [df[i:i+batch_size]['success'].mean() * 100 
                  for i in range(0, len(df), batch_size)]
        ax1.bar(range(len(batches)), batches, color='#3498db', alpha=0.7)
        ax1.axhline(df['success'].mean() * 100, color='r', linestyle='--', 
                   linewidth=2, label=f'Overall: {df["success"].mean()*100:.1f}%')
        ax1.set_xlabel('Episode Batch')
        ax1.set_ylabel('Success Rate (%)')
        ax1.set_title('Success Rate Evolution')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 2. Efficiency Distribution
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(df[df['success']]['efficiency'].to_numpy(), bins=20, alpha=0.7, 
                color='green', label='Success', edgecolor='black')
        ax2.hist(df[~df['success']]['efficiency'].to_numpy(), bins=20, alpha=0.7, 
                color='red', label='Failure', edgecolor='black')
        ax2.set_xlabel('Path Efficiency')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Efficiency Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Box plots comparison
        ax3 = fig.add_subplot(gs[0, 2])
        success_steps = df[df['success']]['steps'].values
        fail_steps = df[~df['success']]['steps'].values
        
        box_data = [success_steps, fail_steps]
        bp = ax3.boxplot(box_data, labels=['Success', 'Failure'],
                        patch_artist=True,
                        boxprops=dict(facecolor='lightblue', alpha=0.7),
                        medianprops=dict(color='red', linewidth=2))
        ax3.set_ylabel('Steps Taken')
        ax3.set_title('Steps Distribution')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Correlation heatmap
        ax4 = fig.add_subplot(gs[1, 0])
        corr_cols = ['efficiency', 'steps', 'total_reward', 'path_smoothness', 'jerk']
        corr_matrix = df[corr_cols].corr()
        
        im = ax4.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        ax4.set_xticks(range(len(corr_cols)))
        ax4.set_yticks(range(len(corr_cols)))
        ax4.set_xticklabels([c.replace('_', '\n') for c in corr_cols], rotation=45)
        ax4.set_yticklabels([c.replace('_', ' ').title() for c in corr_cols])
        ax4.set_title('Metric Correlations')
        
        # Add correlation values
        for i in range(len(corr_cols)):
            for j in range(len(corr_cols)):
                text = ax4.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                              ha="center", va="center", color="black", fontsize=9)
        
        plt.colorbar(im, ax=ax4, label='Correlation')
        
        # 5. Time series with moving average
        ax5 = fig.add_subplot(gs[1, 1])
        window = 5
        rolling_reward = df['total_reward'].rolling(window=window).mean()
        
        ax5.scatter(range(len(df)), df['total_reward'].to_numpy(), alpha=0.3, s=20, label='Raw')
        ax5.plot(range(len(df)), rolling_reward.to_numpy(), 'r-', linewidth=2, 
                label=f'{window}-Episode MA')
        ax5.set_xlabel('Episode')
        ax5.set_ylabel('Total Reward')
        ax5.set_title('Reward Trend Analysis')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Performance radar chart
        ax6 = fig.add_subplot(gs[1, 2], projection='polar')
        
        categories = ['Success\nRate', 'Efficiency', 'Speed', 'Smoothness', 'Safety']
        
        # Normalize metrics to 0-1 scale
        values = [
            df['success'].mean(),
            df['efficiency'].mean(),
            1 - (df['steps'].mean() / df['steps'].max()),
            1 - (df['path_smoothness'].mean() / df['path_smoothness'].max()),
            1 - (df['jerk'].mean() / df['jerk'].max())
        ]
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values += values[:1]
        angles += angles[:1]
        
        ax6.plot(angles, values, 'o-', linewidth=2, label='Agent Performance')
        ax6.fill(angles, values, alpha=0.25)
        ax6.set_xticks(angles[:-1])
        ax6.set_xticklabels(categories)
        ax6.set_ylim(0, 1)
        ax6.set_title('Overall Performance Profile', pad=20)
        ax6.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax6.grid(True)
        
        plt.suptitle('Comprehensive Performance Comparison', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        if save:
            save_path = os.path.join(self.save_dir, 'performance_comparison.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f" Performance comparison saved to: {save_path}")
        
        plt.show()


def load_model(model_path):
    # Load trained model[using PPO]
    try:
        return PPO.load(model_path)
    except:
        try:
            return SAC.load(model_path)
        except Exception as e:
            raise Exception(f"Failed to load model: {e}") 
        

def decompose_reward(env, obs, action, next_obs, info):
    # Decompose reward into components
    components = {}
    
    goal_distance = info.get('goal_distance', 0)
    prev_dist = getattr(env, '_prev_goal_dist', goal_distance)
    
    components['distance_reward'] = (prev_dist - goal_distance) * 50.0
    components['forward_bonus'] = 2.0 if action[0] > 0.05 else -1.0
    components['angular_penalty'] = -abs(action[1]) * 0.1
    components['proximity_reward'] = 10.0 if goal_distance < 0.5 else (5.0 if goal_distance < 1.0 else 0.0)
    components['time_penalty'] = -0.01
    
    if goal_distance < env.goal_threshold:
        components['goal_bonus'] = 500.0
    else:
        components['goal_bonus'] = 0.0
    
    if np.min(env.scan_data) < env.collision_threshold:
        components['collision_penalty'] = -100.0
    else:
        components['collision_penalty'] = 0.0
    
    env._prev_goal_dist = goal_distance
    
    return components




def test_with_performance_analysis(model_path, num_episodes=50, slow_motion=False, 
                              save_analysis=True, visualize=True, use_moving_obstacles=False):
   # Final Report evaluation start>>>>>
    
    rclpy.init()
    
    print(f"Loading model from {model_path}")
    model = load_model(model_path)
    
    env = RLGoalNavTB3Env()
    marker_pub = EnhancedVisualMarkerPublisher()
    analyzer = PerformanceAnalyzer()
    
    print("\n" + "="*100)
    print(" " * 20 + "FINAL REPORT EVALUATION")
    print("="*100)

    if use_moving_obstacles:
        print(" ### Dynamic Moving Obstacles ###")
    print("="*100 + "\n")
    
    # Setup moving obstacles [Hardcoded]
    obstacles = []
    if use_moving_obstacles:
        obstacles = [
            MovingObstacle(1.0, 1.0, 0.2, 0.15),
            MovingObstacle(-1.0, -1.0, -0.15, 0.2),
            MovingObstacle(0.5, -1.5, 0.1, -0.1)
        ]
        for obs in obstacles:
            marker_pub.add_moving_obstacle(obs)
        print(f" Added {len(obstacles)} moving obstacles")
    
    try:
        for episode in range(num_episodes):
            print(f"\n{'='*100}")
            print(f"Episode {episode + 1}/{num_episodes}")
            print('='*100)
            
            obs, _ = env.reset()
            
            goal_x = float(env.goal_position[0])
            goal_y = float(env.goal_position[1])
            marker_pub.spawn_goal_sphere(goal_x, goal_y, episode_num=episode)
            
            print(f" Goal: ({goal_x:.2f}, {goal_y:.2f})")
            print(f" Start: (0.00, 0.00)")
            print(f" Initial Distance: {np.linalg.norm(env.goal_position - env.position):.2f}m\n")
            
            done = False
            truncated = False
            episode_reward = 0
            steps = 0
            trajectory = []
            
            while not done and not truncated:
                trajectory.append(env.position.copy())
                
                action, _ = model.predict(obs, deterministic=True)
                
                rclpy.spin_once(env, timeout_sec=0.01)
                rclpy.spin_once(marker_pub, timeout_sec=0.01)
                
                next_obs, reward, done, truncated, info = env.step(action)
                
                # Record comprehensive data
                analyzer.record_step(episode, steps, obs, action, reward, info)
                analyzer.record_attention(episode, steps, env.scan_data, env.position, env.goal_position)
                analyzer.record_policy_rollout(episode, steps, obs, action)
                
                reward_components = decompose_reward(env, obs, action, next_obs, info)
                analyzer.record_reward_components(episode, steps, reward_components)
                
                episode_reward += reward
                steps += 1
                obs = next_obs
                
                if steps % 25 == 0:
                    distance = info.get('goal_distance', 0)
                    min_obs_dist = np.min(env.scan_data)
                    print(f"  Step {steps}: Distance = {distance:.3f}m, Min Obstacle Dist = {min_obs_dist:.3f}m, Reward = {reward:.2f}")
                
                time.sleep(0.05 if slow_motion else 0.01)
            
            success = info.get('success', info.get('goal_distance', 1.0) < env.goal_threshold)
            
            analyzer.record_episode(episode, trajectory, env.goal_position.tolist(), 
                                   success, episode_reward, steps, obstacles)
            
            print(f"\n{'='*100}")
            if success:
                print(f"!!SUCCESS!!")
            else:
                print(f"✗ FAILED ")
            print(f"  Total Steps: {steps}")
            print(f"  Total Reward: {episode_reward:.2f}")
            print(f"  Final Distance: {info.get('goal_distance', 0):.3f}m")
            print(f"  Path Efficiency: {analyzer.episode_data[-1]['efficiency']:.3f}")
            print('='*100)
            
            time.sleep(0.5)
        
        # Generate analysis
        print("\n" + "="*100)
        print(" " * 30 + "GENERATING ANALYSIS")
        print("="*100)
                
        if visualize:            
            print("  → Creating advanced trajectory analysis...")
            analyzer.visualize_advanced_trajectories(save=save_analysis)

            print("  → Creating attention/saliency visualizations...")
            analyzer.visualize_attention_saliency(episode_idx=0, save=save_analysis)

            print("  → Creating reward breakdown analysis...")
            analyzer.visualize_reward_breakdown(save=save_analysis)

                
    except KeyboardInterrupt:
        print("\n⚠️  Evaluation interrupted by user")
        
    finally:
        marker_pub.delete_goal_sphere()
        env.close()
        marker_pub.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    import sys
    
    # Even I get confused how to use this sometimes, haha
    if len(sys.argv) < 2:
        print("Usage: python3 test_agent_enhanced.py <model_path> [num_episodes] [options]")
        sys.exit(1)
    
    model_path = sys.argv[1]
    num_episodes = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 50
    
    test_with_performance_analysis(model_path, num_episodes, False, 
                              save_analysis=True, visualize=True,
                              use_moving_obstacles=False)