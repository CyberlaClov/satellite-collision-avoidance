from environment import SpaceEnv
from sarsa import SarsaAgent
from visualization import Visualizer


def main():
    # Initialize environment and visualization
    env = SpaceEnv()
    vis = Visualizer(env)

    # Create SARSA agent with optimal parameters
    agent = SarsaAgent(
        env=env,
        alpha=0.1,           # Learning rate
        gamma=0.99,          # Discount factor
        epsilon=1.0,         # Initial exploration rate
        epsilon_decay=0.995, # Epsilon decay rate
        min_epsilon=0.01     # Minimum exploration rate
    )

    # Train agent
    print("Starting SARSA training...")
    num_episodes = 100
    agent.train(num_episodes)

    # Generate reward curve visualization
    vis.plot_training_results(agent.episode_rewards)

    # Run and animate a final test episode with the trained agent
    vis.run_visualization_episode(agent)

if __name__ == "__main__":
    main()
