import numpy as np
import matplotlib.pyplot as plt

from agent import QLearningAgent
from environment import StudyEnvironment



# TRAINING FUNCTION
def train(env, agent, episodes=3000):
    rewards = []
    score_history = []

    # ADDED: action tracking
    action_counts = []
    
    for ep in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0

        for _ in range(env.steps_per_episode):
            action = agent.choose_action(state)

            # track actions
            action_counts.append(action)

            next_state, reward, done, _ = env.step(action)

            agent.update(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            if done:
                break

        agent.decay_epsilon()
        rewards.append(total_reward)
        score_history.append(env.scores.copy())

        # Print progress
        if ep % 100 == 0:
            print(f"Episode {ep}, Avg Reward: {np.mean(rewards[-100:]):.2f}")

    return np.array(rewards), np.array(score_history), np.array(action_counts)


# PLOT TRAINING
def plot_rewards(rewards):
    plt.figure(figsize=(10, 4))

    plt.plot(rewards, alpha=0.4, label="Episode Reward")

    # Moving average
    window = 50
    if len(rewards) >= window:
        ma = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(rewards)), ma, linewidth=2, label="Moving Avg")

    plt.title("Training Progress")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True)
    plt.show()


# TEST SIMULATION
def simulate(env, agent):
    print("\n--- Test Week Simulation ---")

    agent.epsilon = 0  # greedy policy
    state = env.reset()

    print(f"Start Scores: {env.scores}")

    week_scores = [env.scores.copy()]

    for day in range(1, env.steps_per_episode + 1):
        action = agent.choose_action(state)

        next_state, reward, done, info = env.step(action)

        print(
            f"Day {day}: Study {info['subject']}, "
            f"Change={info['delta']}, Scores={info['scores']}"
        )

        week_scores.append(info["scores"].copy())
        state = next_state

        if done:
            break

    return np.array(week_scores)


# PLOT TEST WEEK
def plot_test(week_scores, subjects):
    plt.figure(figsize=(8, 4))

    x = np.arange(len(week_scores))

    for i, subj in enumerate(subjects):
        plt.plot(x, week_scores[:, i], marker='o', label=subj)

    plt.title("Test Week Score Progression")
    plt.xlabel("Step")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.show()


# NEW ADDITIONS START HERE

# Performance Metrics
def evaluate_performance(rewards, score_history):
    print("\n--- Performance Evaluation ---")

    avg_reward = np.mean(rewards)
    std_reward = np.std(rewards)

    first_100 = np.mean(rewards[:100])
    last_100 = np.mean(rewards[-100:])

    improvement = ((last_100 - first_100) / abs(first_100)) * 100

    final_scores = np.mean(score_history[-100:], axis=0)

    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Reward Stability (std): {std_reward:.2f}")
    print(f"Improvement over training: {improvement:.2f}%")

    print("\nFinal Average Scores:")
    print(f"Math: {final_scores[0]:.2f}")
    print(f"Physics: {final_scores[1]:.2f}")
    print(f"Chemistry: {final_scores[2]:.2f}")


# Score Trend Graph
def plot_score_history(score_history, subjects):
    plt.figure(figsize=(10, 4))

    for i, subj in enumerate(subjects):
        plt.plot(score_history[:, i], label=subj)

    plt.title("Score Trend Across Training")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.show()


# Action Distribution
def plot_action_distribution(actions):
    plt.figure(figsize=(6, 4))

    plt.hist(actions, bins=[-0.5, 0.5, 1.5, 2.5], rwidth=0.8)
    plt.xticks([0, 1, 2], ["Math", "Physics", "Chemistry"])

    plt.title("Action Distribution")
    plt.xlabel("Subject")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()


# MAIN
def main():
    # Create environment
    env = StudyEnvironment(steps_per_episode=20)

    # Create agent
    agent = QLearningAgent(
        n_actions=3,
        alpha=0.25,
        gamma=0.9,
        epsilon=1.0,
        epsilon_decay=0.999,
        epsilon_min=0.1,
    )

    print("Training started...\n")

    rewards, score_history, actions = train(env, agent, episodes=5000)

    print("\nTraining completed.")

    # Plot training
    plot_rewards(rewards)

    # ✅ NEW: performance evaluation
    evaluate_performance(rewards, score_history)

    # ✅ NEW: score trend graph
    plot_score_history(score_history, env.SUBJECTS)

    # ✅ NEW: action distribution
    plot_action_distribution(actions)

    # Test simulation
    week_scores = simulate(env, agent)

    # Plot test results
    plot_test(week_scores, env.SUBJECTS)


if __name__ == "__main__":
    main()