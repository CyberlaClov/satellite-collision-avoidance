import numpy as np
from collections import defaultdict


class FeatureExtractor:
    """Feature extractor to transform raw observations into feature vectors for linear function approximation."""
    
    def __init__(self, env, num_tilings=8, num_tiles=8):
        """Initialize the feature extractor.
        
        Args:
            env: The environment
            num_tilings: Number of tilings to use in the tile coding
            num_tiles: Number of tiles per dimension
        """
        self.env = env
        self.num_tilings = num_tilings
        self.num_tiles = num_tiles
        
        # Define which state features to use - focus more on position and velocity
        self.state_features = [
            1,  # pos_y (orbital position - most critical feature)
            3,  # vel_y (vertical velocity)
            4,  # fuel level (important for long-term planning)
            5,  # debris1 relative_pos_x
            6,  # debris1 relative_pos_y
            10, # debris2 relative_pos_x
            11, # debris2 relative_pos_y
        ]
        
        # Define bounds for each feature
        self.state_bounds = {
            1: (-10.0, 10.0),    # pos_y range - expanded to match environment
            3: (-10.0, 10.0),    # vel_y range - expanded to handle higher velocities
            4: (0.0, 5.0),       # fuel level - matches initial fuel amount
            5: (0.0, 5.0),       # relative_pos_x debris 1
            6: (-3.0, 3.0),      # relative_pos_y debris 1
            10: (0.0, 5.0),      # relative_pos_x debris 2
            11: (-3.0, 3.0),     # relative_pos_y debris 2
        }
        
        # Calculate the feature vector size
        self.num_features = len(self.state_features) * num_tilings

    def get_features(self, observation):
        """Convert an observation into a feature vector using tile coding.
        
        Args:
            observation: The raw observation from the environment
            
        Returns:
            A feature vector (sparse representation as list of active indices)
        """
        active_features = []
        
        # Extract relevant features from observation
        state_values = [observation[i] if i < len(observation) else 0 for i in self.state_features]
        
        # For each feature dimension
        for i, feature_idx in enumerate(self.state_features):
            # Get bounds for normalization
            low, high = self.state_bounds.get(feature_idx, (-1.0, 1.0))
            
            # Clip value to be within bounds
            value = max(low, min(high, state_values[i]))
            
            # Normalize to [0, 1]
            normalized = (value - low) / (high - low) if high > low else 0.5
            
            # Special handling for orbital position (pos_y) - use finer tiles near the center
            if feature_idx == 1:  # pos_y
                # Create more tiles around the center (desired orbit at y=0)
                # by distorting the normalized value to concentrate resolution near 0.5
                if normalized < 0.5:
                    # Map [0, 0.5] to [0, 0.5] with higher resolution near 0.5
                    normalized = normalized**0.7  # Power < 1 concentrates resolution near upper end
                else:
                    # Map [0.5, 1] to [0.5, 1] with higher resolution near 0.5
                    normalized = 1 - ((1 - normalized)**0.7)
            
            # For each tiling
            for t in range(self.num_tilings):
                # Add tiling offset (different for each tiling)
                offset = t / self.num_tilings
                
                # Calculate tile index for this feature in this tiling
                tile_idx = int((normalized + offset) % 1.0 * self.num_tiles)
                
                # Calculate the unique index for this tile in the weight vector
                feature_index = (i * self.num_tilings + t) * self.num_tiles + tile_idx
                
                # Add to active features
                active_features.append(feature_index)
        
        return active_features


class SemiGradientSarsaAgent:
    """Semi-gradient SARSA agent with linear function approximation for continuous states."""
    
    def __init__(
        self,
        env,
        alpha=0.01,         # Learning rate (step size)
        gamma=0.99,         # Discount factor
        epsilon=0.1,        # Initial exploration rate
        epsilon_decay=0.995, # Epsilon decay rate
        min_epsilon=0.01,   # Minimum exploration rate
        num_tilings=8,      # Number of tilings for tile coding
        num_tiles=8,        # Number of tiles per dimension
        logger=None         # Optional logger
    ):
        """Initialize the semi-gradient SARSA agent.
        
        Args:
            env: The environment to interact with
            alpha: Learning rate (step size)
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_decay: Epsilon decay rate
            min_epsilon: Minimum exploration rate
            num_tilings: Number of tilings for tile coding
            num_tiles: Number of tiles per dimension 
            logger: Optional logger for visualization
        """
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
        
        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor(
            env, 
            num_tilings=num_tilings, 
            num_tiles=num_tiles
        )
        
        # Calculate maximum number of features
        max_features = len(self.feature_extractor.state_features) * num_tilings * num_tiles
        
        # Initialize weights for each action with small random values
        # Using small random values helps break symmetry in initial policy
        self.weights = np.random.uniform(-0.001, 0.001, (self.num_actions, max_features))
        
        # Episode history
        self.episode_rewards = []
        self.episode_lengths = []
    
    def get_q_value(self, features, action):
        """Calculate Q-value for given features and action using linear function approximation.
        
        Args:
            features: List of active feature indices
            action: The action to evaluate
            
        Returns:
            The estimated Q-value
        """
        action_idx = np.where(self.actions == action)[0][0]
        
        # Linear function approximation: Q(s,a) = sum(w_i * x_i)
        q_value = 0
        for feature_idx in features:
            q_value += self.weights[action_idx, feature_idx]
        
        return q_value
    
    def select_action(self, observation):
        """Select action using epsilon-greedy policy.
        
        Args:
            observation: The current observation
            
        Returns:
            Selected action
        """
        # Extract features from observation
        features = self.feature_extractor.get_features(observation)
        
        # Exploration: choose a random action
        if np.random.random() < self.epsilon:
            return np.random.choice(self.actions)
        
        # Exploitation: choose best action according to Q-values
        q_values = [self.get_q_value(features, action) for action in self.actions]
        best_action_idx = np.argmax(q_values)
        return self.actions[best_action_idx]
    
    def update_weights(self, features, action, target, prediction):
        """Update weights using semi-gradient method.
        
        Args:
            features: List of active feature indices
            action: The action taken
            target: Target Q-value
            prediction: Current predicted Q-value
        """
        action_idx = np.where(self.actions == action)[0][0]
        
        # Calculate the TD error
        td_error = target - prediction
        
        # Update weights for each active feature
        for feature_idx in features:
            # Semi-gradient update: w_i += alpha * TD_error * x_i
            # For tile coding, x_i is 1 for active tiles, 0 otherwise
            self.weights[action_idx, feature_idx] += self.alpha * td_error
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
    
    def train(self, num_episodes):
        """Train the agent using semi-gradient SARSA algorithm.
        
        Args:
            num_episodes: Number of episodes to train for
        """
        for episode in range(num_episodes):
            # Start logging this episode if logger is available
            if self.logger:
                self.logger.start_episode()
            
            # Reset environment and get initial state
            observation = self.env.reset()
            
            # Get features for the initial state
            features = self.feature_extractor.get_features(observation)
            
            # Choose initial action
            action = self.select_action(observation)
            
            total_reward = 0
            done = False
            steps = 0
            
            # Episode loop
            while not done:
                # Take step in environment
                next_observation, reward, done, info = self.env.step(action)
                
                # Log step if logger is available
                if self.logger:
                    self.logger.log_step(action, reward, info)
                
                # Get features for next state
                next_features = self.feature_extractor.get_features(next_observation)
                
                # Get current Q-value prediction
                prediction = self.get_q_value(features, action)
                
                # Initialize target with reward
                target = reward
                
                # If not terminal state, add discounted next Q-value
                if not done:
                    # Choose next action using policy
                    next_action = self.select_action(next_observation)
                    
                    # Calculate next Q-value
                    next_q = self.get_q_value(next_features, next_action)
                    
                    # Add discounted next Q-value to target
                    target += self.gamma * next_q
                    
                    # Prepare for next iteration
                    action = next_action
                
                # Update weights
                self.update_weights(features, action, target, prediction)
                
                # Update state features
                features = next_features
                
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
            self.decay_epsilon()
            
            # Print episode results
            print(
                f"Episode {episode+1}/{num_episodes}, "
                f"Reward: {total_reward:.2f}, "
                f"Steps: {steps}, "
                f"Epsilon: {self.epsilon:.4f}"
            )