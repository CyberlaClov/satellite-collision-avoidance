import time

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.patches import Arrow, Circle

matplotlib.use('TkAgg')  # Use TkAgg backend for interactive animation

class Visualizer:
    def __init__(self, env):
        self.env = env
        self.history = {
            'satellite_x': [],
            'satellite_y': [],
            'satellite_vel_y': [],
            'satellite_fuel': [],
            'time': [],
            'reward': [],
            'debris_positions': [],
            'collision_probs': []
        }
        self.animation = None
        self.fig = None
        self.ax = None
        self.reward_ax = None

        # Colors for better visualization
        self.colors = {
            'satellite': '#1E88E5',     # Vibrant blue for satellite
            'trajectory': '#90CAF9',    # Light blue for satellite trajectory
            'orbit': '#4CAF50',         # Green for planned orbit
            'debris': '#F44336',        # Red for debris
            'warning': '#FFC107',       # Amber for warnings
            'fuel': '#FF9800',          # Orange for fuel indicator
            'background': '#F5F5F5',    # Light gray background
            'grid': '#E0E0E0'           # Subtle grid lines
        }

    def reset(self):
        """Reset visualization history"""
        self.history = {
            'satellite_x': [self.env.pos_x],
            'satellite_y': [self.env.pos_y],
            'satellite_vel_y': [self.env.vel_y],
            'satellite_fuel': [self.env.fuel],
            'time': [0],
            'reward': [0],
            'debris_positions': [[(d.pos_x, d.pos_y) for d in self.env.debris_list]],
            'collision_probs': [[self.env._get_collision_probability(d) for d in self.env.debris_list]]
        }

    def update(self):
        """Update visualization history with current environment state"""
        self.history['satellite_x'].append(self.env.pos_x)
        self.history['satellite_y'].append(self.env.pos_y)
        self.history['satellite_vel_y'].append(self.env.vel_y)
        self.history['satellite_fuel'].append(self.env.fuel)
        self.history['time'].append(self.env.time)

        # Get the latest reward (approximate using the environment's reward function)
        current_reward = self.env.get_reward()
        self.history['reward'].append(current_reward)

        # Store debris positions and collision probabilities
        debris_pos = [(d.pos_x, d.pos_y) for d in self.env.debris_list]
        self.history['debris_positions'].append(debris_pos)

        collision_probs = [self.env._get_collision_probability(d) for d in self.env.debris_list]
        self.history['collision_probs'].append(collision_probs)

    def create_animation(self, filename='satellite_animation.mp4', fps=15):
        """Create enhanced animation from recorded history with reward panel"""
        # Create figure with GridSpec for layout
        fig = plt.figure(figsize=(12, 8))
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])

        # Main simulation display
        ax_main = fig.add_subplot(gs[0])
        ax_main.set_facecolor(self.colors['background'])

        # Reward plot below
        ax_reward = fig.add_subplot(gs[1])
        ax_reward.set_facecolor(self.colors['background'])

        # Calculate the needed padding based on satellite trajectory
        min_x = min(self.history['satellite_x']) - 3
        max_x = max(self.history['satellite_x']) + 10

        def init():
            ax_main.clear()
            ax_main.grid(True, color=self.colors['grid'], linestyle='-', linewidth=0.5, alpha=0.6)
            ax_reward.clear()
            ax_reward.grid(True, color=self.colors['grid'], linestyle='-', linewidth=0.5, alpha=0.6)
            return []

        def update(frame):
            # Clear axes for fresh drawing
            ax_main.clear()

            # Get the data for this frame
            sat_x = self.history['satellite_x'][frame]
            sat_y = self.history['satellite_y'][frame]
            vel_y = self.history['satellite_vel_y'][frame]
            fuel = self.history['satellite_fuel'][frame]
            current_time = self.history['time'][frame]
            debris_pos = self.history['debris_positions'][frame]
            collision_probs = self.history['collision_probs'][frame] if frame < len(self.history['collision_probs']) else []

            # Draw the satellite's trajectory up to this point
            if frame > 1:
                # Create trajectory line with color gradient based on time
                points = np.array([self.history['satellite_x'][:frame],
                                   self.history['satellite_y'][:frame]]).T.reshape(-1, 1, 2)
                segments = np.concatenate([points[:-1], points[1:]], axis=1)

                # Create a line collection for efficient plotting
                lc = LineCollection(segments, cmap='Blues', linewidth=2, alpha=0.7)
                lc.set_array(np.linspace(0, 1, len(segments)))
                ax_main.add_collection(lc)

            # Plot satellite with size indicating fuel level
            fuel_size = 80 + (fuel/self.env.initial_fuel) * 100
            satellite = ax_main.scatter(sat_x, sat_y, s=fuel_size,
                                       color=self.colors['satellite'],
                                       edgecolor='white', linewidth=1.5,
                                       label='Satellite', zorder=10)

            # Add velocity vector
            if abs(vel_y) > 0.1:
                vector_scale = 0.5
                vel_arrow = Arrow(sat_x, sat_y, 0, vel_y * vector_scale,
                                 width=0.3, color=self.colors['satellite'], alpha=0.8)
                ax_main.add_patch(vel_arrow)

            # Plot debris with size indicating collision probability
            debris_plots = []
            for i, pos in enumerate(debris_pos):
                prob = collision_probs[i] if i < len(collision_probs) else 0
                # Size and color based on collision probability
                size = 30 + prob * 1000  # Scale up for visibility
                alpha = min(0.3 + prob * 10, 1.0)  # More opacity for dangerous debris

                debris = ax_main.scatter(pos[0], pos[1], s=size,
                                        color=self.colors['debris'], alpha=alpha,
                                        edgecolor='white', linewidth=0.5)
                debris_plots.append(debris)

                # Add danger radius for high probability debris
                if prob > 0.001:
                    danger_radius = (-np.log(prob/0.005) * 4.9 / np.log(1000) + 0.1)
                    danger_circle = Circle(pos, danger_radius, fill=False,
                                          color=self.colors['warning'],
                                          linestyle='--', alpha=min(prob*200, 0.8))
                    ax_main.add_patch(danger_circle)

                    # Add probability text for significant dangers
                    if prob > 0.002:
                        ax_main.text(pos[0], pos[1] + 0.2, f"{prob:.4f}",
                                    fontsize=8, ha='center', color='black',
                                    bbox=dict(facecolor='white', alpha=0.7, pad=1))

            # Plot planned orbit
            x_range = np.linspace(min_x, max_x, 100)
            orbit = ax_main.plot(x_range, np.zeros_like(x_range),
                                linestyle='--', linewidth=2,
                                color=self.colors['orbit'],
                                label='Planned Orbit', zorder=5)[0]

            # Add information panel
            info_text = (
                f"Time: {current_time:.1f}s\n"
                f"Fuel: {fuel:.2f}\n"
                f"Velocity Y: {vel_y:.2f}\n"
                f"Deviation: {abs(sat_y):.2f}"
            )
            ax_main.text(0.02, 0.98, info_text, transform=ax_main.transAxes,
                        verticalalignment='top', horizontalalignment='left',
                        bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'),
                        fontsize=10)

            # Set axis properties
            ax_main.set_xlim(sat_x - 2, sat_x + 10)
            ax_main.set_ylim(-3, 3)
            ax_main.set_xlabel("X Position")
            ax_main.set_ylabel("Y Position")
            ax_main.set_title(f"Satellite Navigation - Time: {current_time:.1f}")
            ax_main.grid(True, color=self.colors['grid'], linestyle='-', linewidth=0.5, alpha=0.6)
            ax_main.legend(loc='upper right')

            # Update reward plot
            if frame > 0:
                time_values = self.history['time'][:frame+1]
                reward_values = self.history['reward'][:frame+1]

                ax_reward.clear()
                ax_reward.plot(time_values, reward_values, color='#2E7D32', linewidth=2)
                ax_reward.set_xlabel("Time")
                ax_reward.set_ylabel("Reward")
                ax_reward.set_title("Instantaneous Reward")
                ax_reward.grid(True, color=self.colors['grid'], linestyle='-', linewidth=0.5, alpha=0.6)

            return [satellite, orbit] + debris_plots

        # Create the animation
        anim = FuncAnimation(fig, update, frames=len(self.history['satellite_x']),
                             init_func=init, blit=False, interval=1000/fps)

        # Save with higher quality
        anim.save(filename, writer='ffmpeg', fps=fps, dpi=150)
        plt.close(fig)
        print(f"Animation saved as '{filename}'")

    def plot_training_results(self, rewards, lengths=None, window_size=10):
        """Create an enhanced visualization of training rewards"""
        plt.figure(figsize=(12, 6))
        plt.style.use('ggplot')

        # Background and grid styling
        ax = plt.gca()
        ax.set_facecolor(self.colors['background'])

        # Plot raw reward data with low opacity
        plt.plot(rewards, 'o', color='#90CAF9', alpha=0.3, markersize=4, label='Episode Rewards')

        # Calculate and plot moving average with a thicker line
        if len(rewards) >= window_size:
            moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
            x_values = range(window_size-1, len(rewards))
            plt.plot(x_values, moving_avg, color='#1976D2', linewidth=3,
                     label=f'Moving Average (window={window_size})')

            # Add trend line (simple linear regression)
            z = np.polyfit(x_values, moving_avg, 1)
            p = np.poly1d(z)
            plt.plot(x_values, p(x_values), "r--", linewidth=1, alpha=0.7,
                     label=f'Trend (slope: {z[0]:.4f})')

        # Mark best episode
        if len(rewards) > 0:
            best_idx = np.argmax(rewards)
            best_reward = rewards[best_idx]
            plt.scatter([best_idx], [best_reward], s=150, color='#4CAF50',
                      edgecolor='white', linewidth=2, zorder=10,
                      label=f'Best Reward: {best_reward:.2f}')

        # Styling
        plt.title('Training Progress', fontsize=16, pad=20)
        plt.xlabel('Episode', fontsize=12, labelpad=10)
        plt.ylabel('Total Reward', fontsize=12, labelpad=10)

        # Add grid with custom styling
        plt.grid(True, linestyle='--', alpha=0.7)

        # Customize legend
        plt.legend(loc='lower right', frameon=True, framealpha=0.95)

        # Add annotations
        if len(rewards) > 5:
            last_5_avg = np.mean(rewards[-5:])
            plt.figtext(0.01, 0.01,
                      f"Last 5 episodes avg: {last_5_avg:.2f}\n"
                      f"Total episodes: {len(rewards)}",
                      bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'),
                      fontsize=10)

        plt.tight_layout()
        plt.savefig('sarsa_training_results.png', dpi=300)
        plt.show()

    def run_visualization_episode(self, agent, exploration_rate=0.05, animation_delay=0.05):
        """Run a visualization episode with the trained agent"""
        print("\nRunning test episode with the trained policy...")

        # Reset environment and visualizer
        observation = self.env.reset()
        self.reset()

        # Setup for data collection only (no real-time visualization)
        total_reward = 0
        state = agent.discretize_state(observation)
        done = False

        try:
            while not done:
                # Select action with minimal exploration
                if np.random.random() > exploration_rate:
                    action = agent.select_action(state)
                else:
                    action = np.random.choice(self.env.action_space)

                # Take step
                observation, reward, done, info = self.env.step(action)

                # Update data for visualization
                self.update()

                # Accumulate reward
                total_reward += reward

                # Get next state
                state = agent.discretize_state(observation)

                # Print step info occasionally
                if self.env.steps % 50 == 0:
                    print(f"Step: {self.env.steps}, Total Reward: {total_reward:.2f}")

            print(f"\nEpisode finished! Reason: {info['done_reason']}")
            print(f"Total reward: {total_reward:.2f}")
            print(f"Episode length: {self.env.steps} steps")

            # Generate high-quality animation from collected data
            print("\nGenerating animation, please wait...")
            self.create_animation("sarsa_animation.mp4", fps=30)

        except KeyboardInterrupt:
            print("\nSimulation interrupted by user")
            # Generate animation of completed part
            self.create_animation("sarsa_interrupted_animation.mp4")
