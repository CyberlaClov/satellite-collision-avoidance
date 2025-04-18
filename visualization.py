import os
from datetime import datetime

import numpy as np
import rerun as rr
import matplotlib.pyplot as plt


class RerunLogger:
    """Minimal working Rerun logger for satellite collision avoidance."""

    def __init__(
        self,
        env,
        experiment_name=None,
        log_frequency=50,
        logging_mode="viewer",
        log_dir="rerun_logs",
    ):
        """Initialize the Rerun logger.

        Args:
            env: The environment being logged
            experiment_name: Optional name for the experiment
            log_frequency: How often to log full episodes
            logging_mode: One of "viewer", "file", or "both"
            log_dir: Directory to store log files when file logging is enabled
        """
        self.env = env

        # Create a unique timestamp for logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create a unique experiment name with timestamp if none provided
        if experiment_name is None:
            experiment_name = f"satellite_collision_avoidance_{timestamp}"

        self.experiment_name = experiment_name
        self.log_frequency = log_frequency
        self.logging_mode = logging_mode
        self.log_dir = log_dir

        # Initialize Rerun
        rr.init(self.experiment_name)

        # Handle different logging modes
        if logging_mode == "viewer":
            rr.spawn()

        elif logging_mode == "file":
            # Create log directory if it doesn't exist
            os.makedirs(log_dir, exist_ok=True)

            # Create log file with timestamp
            log_filename = os.path.join(log_dir, f"satellite_log_{timestamp}.rrd")
            rr.save(log_filename)
            self.log_filename = log_filename

        # Track episode counter for periodic full episode logging
        self.episode_counter = 0
        self.logging_full_episode = False
        
        # For learning curve plotting
        self.episode_rewards_history = []
        self.moving_avg_window = 100  # Window size for moving average

    def start_episode(self):
        """Start tracking a new episode"""
        self.episode_counter += 1
        self.logging_full_episode = self.episode_counter % self.log_frequency == 0

        if self.logging_full_episode:
            # Reset past positions for new episode
            self.past_positions = []

            # Log episode start
            rr.set_time_sequence(f"episode_{self.episode_counter}", 0)

            # Add basic info about the environment
            self._log_orbit()

    def _log_orbit(self):
        """Log the planned orbit as a line"""
        # Create line segments for the planned orbit
        x_vals = np.linspace(-5, 15, 2)  # Just two points for a straight line
        y_vals = np.zeros_like(x_vals)
        segments = [
            (np.array([x_vals[i], y_vals[i]]), np.array([x_vals[i + 1], y_vals[i + 1]]))
            for i in range(len(x_vals) - 1)
        ]

        # Log the orbit line
        rr.log(
            "world/orbit",
            rr.LineStrips2D(segments, radii=0.03, colors=[0, 255, 0, 128]),
        )

    def log_step(self, action, reward, info=None):
        """Log a single step of the environment."""
        if not self.logging_full_episode:
            return

        # Set the time for this step
        rr.set_time_sequence(f"episode_{self.episode_counter}", self.env.steps)

        # Log the satellite position
        self._log_satellite()

        self._log_action(action)

        # Log debris
        self._log_debris()

        # Log metrics
        self._log_metrics(reward)

    def _log_satellite(self):
        """Log the satellite position"""
        # Log satellite as a point with increased size
        satellite_pos = np.array([self.env.pos_x, self.env.pos_y])
        rr.log(
            "world/satellite",
            rr.Points2D(
                [satellite_pos], radii=0.15, colors=[[0, 100, 255, 255]]
            ),  # Increased from 0.05 to 0.15
        )

        # Store position for trajectory
        if hasattr(self, "past_positions"):
            self.past_positions.append(satellite_pos.copy())
        else:
            self.past_positions = [satellite_pos.copy()]

        # Log all positions from this episode as Points2D
        rr.log(
            "world/trajectory",
            rr.Points2D(self.past_positions, radii=0.03, colors=[[0, 100, 255, 128]]),
        )

    def _log_action(self, action):

        current_pos = np.array([self.env.pos_x, self.env.pos_y])
        action_vec = np.array([0, action])

        rr.log("world/action", rr.Arrows2D(origins=current_pos, vectors=action_vec))

    def _log_debris(self):
        """Log debris objects"""
        # Log each debris as a point with increased size
        debris_positions = [np.array([d.pos_x, d.pos_y]) for d in self.env.debris_list]
        if debris_positions:  # Only log if there are debris
            rr.log(
                "world/debris",
                rr.Points2D(
                    debris_positions, radii=0.075, colors=[[255, 50, 0, 255]]
                ),  # Increased from 0.05 to 0.15
            )

    def _log_metrics(self, reward):
        """Log various metrics as timeseries data"""
        # Log reward
        rr.log("metrics/reward", rr.Scalar(reward))

        # Log fuel
        rr.log("metrics/fuel", rr.Scalar(self.env.fuel))

        # Log y deviation
        rr.log("metrics/y_deviation", rr.Scalar(abs(self.env.pos_y)))

        # Log survival probability
        rr.log(
            "metrics/survival_probability",
            rr.Scalar(self.env.get_survival_probability()),
        )

    def log_episode_summary(self, total_reward, steps, epsilon=None, info=None):
        """Log summary information at the end of an episode."""
        # Log episode summary metrics
        rr.set_time_sequence("full_training", self.episode_counter)
        rr.log("training/episode_reward", rr.Scalar(total_reward))
        rr.log("training/episode_length", rr.Scalar(steps))

        if epsilon is not None:
            rr.log("training/epsilon", rr.Scalar(epsilon))
            
        # Store reward for learning curve plotting
        self.episode_rewards_history.append(total_reward)
        
    def plot_learning_curve(self, save_path=None):
        """Plot the learning curve showing episode rewards over time.
        
        Args:
            save_path: Optional path to save the plot image. If None, will save to log_dir.
        
        Returns:
            Path to the saved image file.
        """
        if not self.episode_rewards_history:
            print("No training data available to plot")
            return None
            
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Plot raw rewards
        episodes = range(1, len(self.episode_rewards_history) + 1)
        plt.plot(episodes, self.episode_rewards_history, alpha=0.3, color='blue', label='Raw rewards')
        
        # Plot moving average if we have enough data
        if len(self.episode_rewards_history) >= self.moving_avg_window:
            moving_avgs = []
            for i in range(len(self.episode_rewards_history) - self.moving_avg_window + 1):
                window_avg = np.mean(self.episode_rewards_history[i:i+self.moving_avg_window])
                moving_avgs.append(window_avg)
            
            # Plot moving average
            plt.plot(
                range(self.moving_avg_window, len(self.episode_rewards_history) + 1),
                moving_avgs,
                color='red',
                linewidth=2,
                label=f'{self.moving_avg_window}-Episode Moving Average'
            )
        
        # Add labels and title
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Learning Curve: Reward vs Episode')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Determine save path
        if save_path is None:
            # Create log directory if it doesn't exist
            os.makedirs(self.log_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.log_dir, f"learning_curve_{timestamp}.png")
        
        # Save figure
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Learning curve saved to: {save_path}")
        return save_path
