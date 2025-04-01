import numpy as np


class SarsaAgent:
    """SARSA (State-Action-Reward-State-Action) agent for reinforcement learning"""

    def __init__(self,
                 env,
                 alpha=0.1,      # Learning rate
                 gamma=0.9,      # Discount factor
                 epsilon=0.3,    # Initial exploration rate
                 epsilon_decay=0.9999,  # Epsilon decay rate
                 min_epsilon=0.01):     # Minimum exploration rate

        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        # Get all possible actions
        self.actions = env.action_space
        self.num_actions = len(self.actions)

        # Define state discretization parameters (which features to use and how many bins)
        self.state_features = [1, 3, 5, 6, 10, 11]  # pos_y, vel_y, debris1 relative x/y, debris2 relative x/y
        self.bins = [20, 10, 10, 10, 10, 10]  # Number of bins for each feature

        # Define observation bounds for normalization
        self.obs_bounds = {
            1: (-3.0, 3.0),       # pos_y range
            3: (-5.0, 5.0),       # vel_y range
            5: (0.0, 10.0),       # relative_pos_x debris 1
            6: (-3.0, 3.0),       # relative_pos_y debris 1
            10: (0.0, 10.0),      # relative_pos_x debris 2
            11: (-3.0, 3.0),      # relative_pos_y debris 2
        }

        # Initialize Q-table with small random values (sparse representation using dictionary)
        self.q_table = {}

        # Episode history
        self.episode_rewards = []
        self.episode_lengths = []

    def discretize_state(self, observation):
        """Convert continuous state to discrete state by binning"""
        discrete_state = []

        for i, feature_idx in enumerate(self.state_features):
            if feature_idx < len(observation):
                # Get bounds for this dimension
                low, high = self.obs_bounds.get(feature_idx, (-10.0, 10.0))

                # Clip the observation value to be within bounds
                val = max(low, min(observation[feature_idx], high))

                # Normalize to [0, 1]
                normalized = (val - low) / (high - low)

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
            action_idx = np.argmax([self.q_table[state][np.where(self.actions == a)[0][0]]
                                  for a in self.actions])
            return self.actions[action_idx]

    def update_q_value(self, state, action, reward, next_state, next_action):
        """Update Q-value using SARSA update rule"""
        # If state not in Q-table, initialize it
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.num_actions)

        # If next_state not in Q-table, initialize it
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(self.num_actions)

        # Get action indices
        action_idx = np.where(self.actions == action)[0][0]
        next_action_idx = np.where(self.actions == next_action)[0][0]

        # Current Q-value
        q_value = self.q_table[state][action_idx]

        # Next Q-value
        next_q_value = self.q_table[next_state][next_action_idx]

        # SARSA update rule: Q(s,a) = Q(s,a) + α * [r + γ * Q(s',a') - Q(s,a)]
        new_q_value = q_value + self.alpha * (reward + self.gamma * next_q_value - q_value)

        # Update Q-table
        self.q_table[state][action_idx] = new_q_value

    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def train(self, num_episodes):
        """Train the agent using SARSA algorithm"""
        for episode in range(num_episodes):
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

            # Decay exploration rate
            self.decay_epsilon()

            # Print episode results
            print(f"Episode {episode}/{num_episodes}, "
                  f"Reward: {total_reward:.2f}, "
                  f"Steps: {steps}, "
                  f"Epsilon: {self.epsilon:.4f}, "
                  f"Q-table size: {len(self.q_table)}")
