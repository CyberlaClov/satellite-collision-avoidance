from environment import SpaceEnv
from sarsa import SarsaAgent
from visualization import RerunLogger


def main():
    # Initialize environment
    env = SpaceEnv()

    # Initialize RerunLogger
    logger = RerunLogger(env, "sarsa_satellite_collision_avoidance", log_frequency=1000)

    # Create SARSA agent with optimal parameters and pass the logger
    agent = SarsaAgent(
        env=env,
        alpha=0.1,  # Learning rate
        gamma=0.99,  # Discount factor
        epsilon=0.5,  # Initial exploration rate
        epsilon_decay=0.995,  # Epsilon decay rate
        min_epsilon=0.01,  # Minimum exploration rate
        logger=logger,  # Pass the logger to the agent
    )

    # Train agent (the agent will use the logger internally)
    print("Starting SARSA training...")
    num_episodes = 10000
    agent.train(num_episodes)

    # No need for visualization code here - the agent handles it internally now
    print("\nProcess complete. View the visualization in the Rerun viewer.")


if __name__ == "__main__":
    main()
