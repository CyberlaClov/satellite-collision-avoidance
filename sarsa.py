import numpy as np


class SarsaAgent:
    """SARSA (State-Action-Reward-State-Action) agent for reinforcement learning"""

    def __init__(
        self,
        env,
        alpha=0.1,  # Learning rate
        gamma=0.9,  # Discount factor
        epsilon=0.3,  # Initial exploration rate
        epsilon_decay=0.9999,  # Epsilon decay rate
        min_epsilon=0.01,  # Minimum exploration rate
        logger=None,
    ):  # Optional RerunLogger

        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.logger = logger

        # Get all possible actions
        self.actions = env.action_space
        self.num_actions = len(self.actions)

        # Define state discretization parameters (which features to use and how many bins)
        self.state_features = [
            1,  # pos_y (orbital position - most critical feature)
            3,  # vel_y (vertical velocity)
            4,  # fuel level (important for long-term planning)
            5,  # debris1 relative_pos_x
            6,  # debris1 relative_pos_y
            10, # debris2 relative_pos_x
            11, # debris2 relative_pos_y
        ]
        # Heavily favor orbit position (y) and velocity (vel_y) by assigning many more bins
        # This makes the agent much more sensitive to changes in orbital position
        self.bins = [20, 15, 4, 4, 4, 4, 4]  # Number of bins for each feature (dramatically increased resolution for pos_y and vel_y)

        # Define observation bounds for normalization
        self.obs_bounds = {
            1: (-10.0, 10.0),  # pos_y range
            3: (-10.0, 10.0),  # vel_y range - expanded to handle higher velocities
            4: (0.0, 5.0),     # fuel level - matches initial fuel amount
            5: (0.0, 5.0),     # relative_pos_x debris 1
            6: (-3.0, 3.0),    # relative_pos_y debris 1
            10: (0.0, 5.0),    # relative_pos_x debris 2
            11: (-3.0, 3.0),   # relative_pos_y debris 2
        }

        # Initialize Q-table with small random values (sparse representation using dictionary)
        self.q_table = {}

        # Episode history
        self.episode_rewards = []
        self.episode_lengths = []

    def discretize_state(self, observation):
        """Convert continuous state to discrete state by binning with improved resolution near orbit"""
        discrete_state = []

        for i, feature_idx in enumerate(self.state_features):
            if feature_idx < len(observation):
                # Get bounds for this dimension
                low, high = self.obs_bounds.get(feature_idx, (-10.0, 10.0))

                # Clip the observation value to be within bounds
                val = max(low, min(observation[feature_idx], high))

                # Normalize to [0, 1]
                normalized = (val - low) / (high - low)
                
                # Special handling for orbital position (pos_y)
                if feature_idx == 1:  # pos_y
                    # Create much more bins around the center (desired orbit at y=0)
                    # by strongly distorting the normalized value to concentrate resolution near 0.5
                    if normalized < 0.5:
                        # Map [0, 0.5] to [0, 0.5] with much higher resolution near 0.5
                        normalized = normalized**0.5  # Lower power means even more concentration near 0.5
                    else:
                        # Map [0.5, 1] to [0.5, 1] with much higher resolution near 0.5
                        normalized = 1 - ((1 - normalized)**0.5)
                
                # Also handle velocity more carefully (feature_idx == 3)
                elif feature_idx == 3:  # vel_y
                    # Create more bins around zero velocity for finer control
                    # This helps the agent learn more precise maneuvers near zero velocity
                    if normalized < 0.5:
                        normalized = normalized**0.6
                    else:
                        normalized = 1 - ((1 - normalized)**0.6)
                
                # Discretize into bins
                bin_index = min(int(normalized * self.bins[i]), self.bins[i] - 1)

                discrete_state.append(bin_index)

        return tuple(discrete_state)

    def select_action(self, state):
        """Select action using epsilon-greedy policy"""
        # Exploration: choose a random action
        if np.random.random() < self.epsilon:
            return np.random.choice(self.actions)

        # Exploitation: choose best action according to Q-table
        else:
            # If state is not in Q-table, initialize it
            if state not in self.q_table:
                self.q_table[state] = np.zeros(self.num_actions)

            # Find action with the highest Q-value
            action_idx = np.argmax(
                [
                    self.q_table[state][np.where(self.actions == a)[0][0]]
                    for a in self.actions
                ]
            )
            return self.actions[action_idx]

    def update_q_value(self, state, action, reward, next_state, next_action):
        """Update Q-value using SARSA update rule with custom shaping for orbit maintenance"""
        # If state not in Q-table, initialize it
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.num_actions)
            
            # Initialize with a slight preference for smaller actions (fuel conservation)
            # and the 0 action (orbit maintenance)
            for i, a in enumerate(self.actions):
                # Add small bias toward action 0 (no maneuver)
                if a == 0:
                    self.q_table[state][i] = 0.05
                # Slightly penalize high-magnitude actions that use more fuel
                else:
                    self.q_table[state][i] = 0.02 - 0.005 * abs(a)

        # If next_state not in Q-table, initialize it
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(self.num_actions)
            
            # Apply same initialization for the next state
            for i, a in enumerate(self.actions):
                if a == 0:
                    self.q_table[next_state][i] = 0.05
                else:
                    self.q_table[next_state][i] = 0.02 - 0.005 * abs(a)

        # Get action indices
        action_idx = np.where(self.actions == action)[0][0]
        next_action_idx = np.where(self.actions == next_action)[0][0]

        # Current Q-value
        q_value = self.q_table[state][action_idx]

        # Next Q-value
        next_q_value = self.q_table[next_state][next_action_idx]

        # SARSA update rule: Q(s,a) = Q(s,a) + α * [r + γ * Q(s',a') - Q(s,a)]
        new_q_value = q_value + self.alpha * (
            reward + self.gamma * next_q_value - q_value
        )

        # Update Q-table
        self.q_table[state][action_idx] = new_q_value

    def decay_epsilon(self, k):
        """Decay exploration rate"""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def train(self, num_episodes):
        """Train the agent using SARSA algorithm with optional logging"""
        for episode in range(num_episodes):
            # Start logging this episode if logger is available
            if self.logger:
                self.logger.start_episode()

            # Reset environment and get initial state
            observation = self.env.reset()
            state = self.discretize_state(observation)

            total_reward = 0
            done = False
            steps = 0

            # Choose initial action
            action = self.select_action(state)

            while not done:
                # Take step in environment
                observation, reward, done, info = self.env.step(action)

                # Log step if logger is available
                if self.logger:
                    self.logger.log_step(action, reward, info)

                next_state = self.discretize_state(observation)

                # Choose next action
                next_action = self.select_action(next_state)

                # Update Q-value
                self.update_q_value(state, action, reward, next_state, next_action)

                # Update state and action
                state = next_state
                action = next_action

                # Accumulate reward and steps
                total_reward += reward
                steps += 1

            # Store episode results
            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(steps)

            # Log episode summary if logger is available
            if self.logger:
                self.logger.log_episode_summary(total_reward, steps, self.epsilon, info)

            # Decay exploration rate
            self.decay_epsilon(k=len(self.episode_rewards))

            # Print episode results
            print(
                f"Episode {episode+1}/{num_episodes}, "
                f"Reward: {total_reward:.2f}, "
                f"Steps: {steps}, "
                f"Epsilon: {self.epsilon:.4f}, "
                f"Q-table size: {len(self.q_table)}"
            )
