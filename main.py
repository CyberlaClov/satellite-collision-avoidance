import os

from environment import SpaceEnv
from sarsa import SarsaAgent
from semigradient_sarsa import SemiGradientSarsaAgent
from visualization import RerunLogger


def main(
    agent_type="sarsa", num_episodes=10000, log_frequency=1000, logging_mode="viewer"
):
    """Main function to train an agent for satellite collision avoidance.

    Args:
        agent_type: Type of agent to use - "sarsa" or "semigradient"
        num_episodes: Number of episodes to train for
        log_frequency: Log visualization every N episodes
        logging_mode: Logging mode for visualizations - "viewer", "file", or "both"
    """
    # Initialize environment
    env = SpaceEnv()

    # Initialize RerunLogger
    experiment_name = f"{agent_type}_satellite_collision_avoidance"
    logger = RerunLogger(
        env,
        experiment_name,
        log_frequency=log_frequency,
        logging_mode=logging_mode,
    )

    # Create agent based on agent_type
    if agent_type == "sarsa":
        agent = SarsaAgent(
            env=env,
            alpha=0.03,  # Learning rate - further reduced for more stable learning
            gamma=0.8,   # Discount factor - significantly reduced to focus heavily on immediate rewards
                         # This will make the agent prioritize orbit maintenance over long-term goals
            epsilon=0.6,  # Initial exploration rate - substantially increased for much wider exploration
            epsilon_decay=0.9997,  # Epsilon decay rate - even slower decay for much longer exploration
            min_epsilon=0.1,  # Minimum exploration rate - higher minimum to keep exploring
            logger=logger,  # Pass the logger to the agent
        )
    elif agent_type == "semigradient":
        agent = SemiGradientSarsaAgent(
            env=env,
            alpha=0.002,  # Learning rate - further reduced for stability
            gamma=0.95,  # Discount factor - slightly reduced to focus more on immediate rewards
            epsilon=0.3,  # Initial exploration rate - increased for better exploration
            epsilon_decay=0.999,  # Epsilon decay rate - even slower decay for longer exploration
            min_epsilon=0.05,  # Minimum exploration rate - increased to maintain some exploration
            num_tilings=16,  # Number of tilings - increased for better generalization
            num_tiles=8,  # Number of tiles per dimension
            logger=logger,  # Pass the logger to the agent
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    # Train agent
    print(f"Starting {agent_type.upper()} training for {num_episodes} episodes...")
    agent.train(num_episodes)

    # Plot and save the learning curve
    learning_curve_path = logger.plot_learning_curve()

    # Print statistics
    if agent.episode_rewards:
        avg_reward = sum(agent.episode_rewards) / len(agent.episode_rewards)
        last_100_avg = sum(agent.episode_rewards[-100:]) / min(
            100, len(agent.episode_rewards)
        )
        print(f"\nTraining Statistics:")
        print(f"Agent Type: {agent_type}")
        print(f"Average Reward: {avg_reward:.2f}")
        print(f"Average Reward (last 100 episodes): {last_100_avg:.2f}")
        print(f"Learning curve saved to: {learning_curve_path}")

    print("\nProcess complete. View the visualization in the Rerun viewer.")


if __name__ == "__main__":
    # Run with default parameters (discrete SARSA)
    # To use semi-gradient SARSA, modify the function call:
    # main(agent_type="semigradient")
    main(agent_type="sarsa")
