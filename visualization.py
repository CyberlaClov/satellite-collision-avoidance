from datetime import datetime

import numpy as np
import rerun as rr


class RerunLogger:
    """Minimal working Rerun logger for satellite collision avoidance."""

    def __init__(self, env, experiment_name=None):
        """Initialize the Rerun logger."""
        self.env = env

        # Create a unique experiment name with timestamp if none provided
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"satellite_collision_avoidance_{timestamp}"

        self.experiment_name = experiment_name

        # Initialize Rerun with spawning the viewer
        rr.init(self.experiment_name, spawn=True)

        # Track episode counter for periodic full episode logging
        self.episode_counter = 0
        self.logging_full_episode = False

    def start_episode(self):
        """Start tracking a new episode"""
        self.episode_counter += 1
        self.logging_full_episode = self.episode_counter % 15 == 0

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
            rr.LineStrips2D(segments, radii=0.01, colors=[0, 255, 0, 128]),
        )

    def log_step(self, action, reward, info=None):
        """Log a single step of the environment."""
        if not self.logging_full_episode:
            return

        # Set the time for this step
        rr.set_time_sequence(f"episode_{self.episode_counter}", self.env.steps)

        # Log the satellite position
        self._log_satellite()

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
                [satellite_pos], radii=0.05, colors=[[0, 0, 255, 255]]
            ),  # Increased from 0.02 to 0.05
        )

        # Store position for trajectory
        if hasattr(self, "past_positions"):
            self.past_positions.append(satellite_pos.copy())
        else:
            self.past_positions = [satellite_pos.copy()]

        # Log all positions from this episode as Points2D
        rr.log(
            "world/trajectory",
            rr.Points2D(self.past_positions, radii=0.015, colors=[[0, 0, 255, 128]]),
        )

    def _log_debris(self):
        """Log debris objects"""
        # Log each debris as a point with increased size
        debris_positions = [np.array([d.pos_x, d.pos_y]) for d in self.env.debris_list]
        if debris_positions:  # Only log if there are debris
            rr.log(
                "world/debris",
                rr.Points2D(
                    debris_positions, radii=0.05, colors=[[255, 0, 0, 255]]
                ),  # Increased from 0.02 to 0.05
            )

    def _log_metrics(self, reward):
        """Log various metrics as timeseries data"""
        # Log reward
        rr.log("metrics/reward", rr.Scalar(reward))

        # Log fuel
        rr.log("metrics/fuel", rr.Scalar(self.env.fuel))

        # Log y deviation
        rr.log("metrics/y_deviation", rr.Scalar(abs(self.env.pos_y)))

    def log_episode_summary(self, total_reward, steps, epsilon=None, info=None):
        """Log summary information at the end of an episode."""
        # Log episode summary metrics
        rr.set_time_sequence("full_training", self.episode_counter)
        rr.log("training/episode_reward", rr.Scalar(total_reward))
        rr.log("training/episode_length", rr.Scalar(steps))

        if epsilon is not None:
            rr.log("training/epsilon", rr.Scalar(epsilon))

    def log_training_progress(self, episode_rewards, episode_lengths, epsilon=None):
        """Log overall training progress metrics."""
        # Only log if we have at least one episode
        if len(episode_rewards) > 0:
            # Log the most recent reward
            latest_reward = episode_rewards[-1]
            rr.set_time_sequence("full_training", len(episode_rewards))
            rr.log("training/latest_reward", rr.Scalar(latest_reward))

            # Log epsilon if provided
            if epsilon is not None:
                rr.log("training/epsilon", rr.Scalar(epsilon))
