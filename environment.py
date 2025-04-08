from typing import Any, Dict, List, Tuple

import numpy as np


class DebrisCluster:
    """Class to represent a single debris cluster"""

    def __init__(self, pos_x, pos_y, vel_x, vel_y):
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.vel_x = vel_x
        self.vel_y = vel_y

    def update(self, dt):
        """Update debris position based on its velocity"""
        self.pos_x += self.vel_x * dt
        self.pos_y += self.vel_y * dt

    def distance_to(self, sat_x, sat_y):
        """Calculate Euclidean distance to the satellite"""
        return np.sqrt((sat_x - self.pos_x) ** 2 + (sat_y - self.pos_y) ** 2)


class SpaceEnv:
    """Environment for satellite collision avoidance"""

    def __init__(
        self,
        pos_x=0.0,
        pos_y=0.0,
        vel_x=0.5,
        vel_y=0.0,
        K=4,  # Number of debris to track
        dt=0.1,
        beta=0.01,  # Uncertainty factor
        max_deviation=2.0,
        p0=5.0,  # Standard operational profit
        initial_fuel=5.0,
        max_steps=2000,
    ):  # 200/dt = 2000 steps

        # Satellite parameters
        self.initial_pos_x = pos_x
        self.initial_pos_y = pos_y
        self.initial_vel_x = vel_x
        self.initial_vel_y = vel_y
        self.initial_fuel = initial_fuel

        # Environment parameters
        self.K = K
        self.dt = dt
        self.beta = beta
        self.max_deviation = max_deviation
        self.p0 = p0
        self.max_steps = max_steps
        self.action_space = np.array([-5, -3, -1, 0, 1, 3, 5])

        # Initialize environment
        self.reset()

    def reset(self):
        """Reset environment to initial state"""
        # Reset satellite state
        self.pos_x = self.initial_pos_x
        self.pos_y = self.initial_pos_y
        self.vel_x = self.initial_vel_x
        self.vel_y = self.initial_vel_y
        self.fuel = self.initial_fuel

        # Reset episode state
        self.time = 0
        self.steps = 0
        self.done = False
        self.debris_list = []

        # Initialize debris generation probability to 0.01
        self.debris_gen_probability = 0.01
        # Time of last debris generation - start at negative dt to get correct initial probability
        self.last_debris_time = -self.dt

        # Generate initial debris clusters
        self._generate_initial_debris()

        # Return initial observation
        return self._get_observation()

    def _generate_initial_debris(self):
        """Generate initial debris clusters as described in section 5.1"""
        # Generate 4 debris clusters
        for _ in range(4):
            # Position uniformly distributed in x ∈ [1, 5], y ∈ [-2, 2]
            pos_x = np.random.uniform(1, 5)
            pos_y = np.random.uniform(-2, 2)

            # Assign collision time between 50Δt and 100Δt
            collision_time = np.random.uniform(50 * self.dt, 100 * self.dt)

            # Calculate velocity for collision at time tn if no evasive action
            # Using equations 12 and 13 from the document
            vel_x = (self.pos_x + self.vel_x * collision_time - pos_x) / collision_time
            vel_y = (self.pos_y + self.vel_y * collision_time - pos_y) / collision_time

            # Create and add debris
            self.debris_list.append(DebrisCluster(pos_x, pos_y, vel_x, vel_y))

    def _check_new_debris(self):
        """Check if new debris should be generated"""
        if np.random.random() < self.debris_gen_probability:
            # Generate new debris at detection boundary (x = S_x + 5)
            pos_x = self.pos_x + 5
            pos_y = np.random.uniform(-2, 2)

            # Assign collision time
            collision_time = np.random.uniform(50 * self.dt, 100 * self.dt)

            # Calculate velocity components (for collision at tn)
            vel_x = (self.pos_x + self.vel_x * collision_time - pos_x) / collision_time
            vel_y = (self.pos_y + self.vel_y * collision_time - pos_y) / collision_time

            # Add the new debris
            self.debris_list.append(DebrisCluster(pos_x, pos_y, vel_x, vel_y))

            # Reset probability to 0 when new debris is generated
            self.debris_gen_probability = 0.0

            # Record time when debris was generated
            self.last_debris_time = self.time

    def _get_collision_probability(self, debris):
        """Calculate collision probability with a debris cluster (equation 15)"""
        distance = debris.distance_to(self.pos_x, self.pos_y)

        if distance <= 0.1:
            return 0.005
        else:
            return 0.005 * np.exp(-np.log(1000) / 4.9 * (distance - 0.1))

    def _get_observation(self):
        """Create observation state vector"""
        # Sort debris by danger (collision probability)
        self.debris_list.sort(
            key=lambda d: self._get_collision_probability(d), reverse=True
        )

        # Keep only K most dangerous debris
        relevant_debris = self.debris_list[: self.K]

        # Create observation vector
        obs = [self.pos_x, self.pos_y, self.vel_x, self.vel_y, self.fuel]

        for debris in relevant_debris:
            collision_prob = self._get_collision_probability(debris)
            relative_pos_x = debris.pos_x - self.pos_x
            relative_pos_y = debris.pos_y - self.pos_y
            obs.extend(
                [
                    relative_pos_x,
                    relative_pos_y,
                    debris.vel_x,
                    debris.vel_y,
                    collision_prob,
                ]
            )

        # Pad observation if fewer than K debris exist
        padding_needed = self.K - len(relevant_debris)
        if padding_needed > 0:
            obs.extend([0.0] * (padding_needed * 5))  # 5 values per debris

        return np.array(obs)

    def get_survival_probability(self):
        """Calculate survival probability based on current state"""
        survival_prob = 1.0
        for debris in self.debris_list:
            collision_prob = self._get_collision_probability(debris)
            survival_prob *= 1 - collision_prob

        return survival_prob

    def get_reward(self):
        """Calculate reward for current state"""
        # Operational profit based on deviation (equation 10)
        orbit_reward = self.p0 * (1 - np.abs(self.pos_y) / self.max_deviation)

        return orbit_reward

    def take_action(self, action):
        """Update environment state based on action"""
        # Ensure action is valid
        if action not in self.action_space:
            raise ValueError(
                f"Invalid action {action}. Valid actions: {self.action_space}"
            )

        # Store previous velocity for position update
        vel_y_prev = self.vel_y

        # Update velocity (equation 6)
        self.vel_y = self.vel_y + action * self.dt

        # Update x position (equation 7)
        self.pos_x = self.pos_x + self.vel_x * self.dt

        # Calculate Gaussian error (equation 9)
        error_std = self.beta * np.abs((self.vel_y + vel_y_prev))
        epsilon = np.random.normal(0, error_std)

        # Update y position with uncertainty (equation 8)
        self.pos_y = self.pos_y + self.dt * (self.vel_y + vel_y_prev) / 2 + epsilon

        # Update fuel
        self.fuel = self.fuel - 0.1 * np.abs(action) * self.dt

        # Update debris positions
        for debris in self.debris_list:
            debris.update(self.dt)

        # Update time
        self.time += self.dt

        # Update debris generation probability
        # Increases linearly from 0 at last_debris_time to 1 at last_debris_time + 100Δt
        time_since_last_debris = self.time - self.last_debris_time
        self.debris_gen_probability = min(time_since_last_debris / (100 * self.dt), 1.0)

        # Check for new debris
        self._check_new_debris()

        # Remove debris that are far behind the satellite
        self.debris_list = [d for d in self.debris_list if d.pos_x > (self.pos_x - 10)]

        self.steps += 1

    def check_collision(self):
        """Check if satellite collides with any debris"""
        for debris in self.debris_list:
            collision_prob = self._get_collision_probability(debris)

            # Sample from the probability to determine if collision occurs
            if np.random.random() < collision_prob:
                return True

        return False

    def is_done(self):
        """Check if episode is done and return reason"""
        # Check for collision
        if self.check_collision():
            return True, "collision"

        # Check fuel depletion
        if self.fuel <= 0:
            return True, "fuel_depletion"

        # Check max steps
        if self.steps >= self.max_steps:
            return True, "max_steps"

        return False, ""

    def step(self, action):
        """Take a step in the environment"""
        # Execute action
        self.take_action(action)

        # Get new observation
        observation = self._get_observation()

        # Calculate reward
        reward = self.get_reward()

        # Check if done
        done, done_reason = self.is_done()
        self.done = done

        # Add termination rewards
        if done:
            if done_reason == "max_steps":
                # Positive reward for reaching max episode length
                reward += 30
            elif done_reason == "fuel_depletion":
                # Negative reward for running out of fuel
                reward -= 30.0
            elif done_reason == "collision":
                # More punishing negative reward for collision
                reward -= 10.0

        # Return step information
        info = {
            "fuel": self.fuel,
            "position": (self.pos_x, self.pos_y),
            "velocity": (self.vel_x, self.vel_y),
            "time": self.time,
            "steps": self.steps,
            "num_debris": len(self.debris_list),
            "done_reason": done_reason if done else "",
            "survival_probability": self.get_survival_probability(),
        }

        return observation, reward, done, info
