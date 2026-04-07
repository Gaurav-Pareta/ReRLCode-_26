import numpy as np


class QLearningAgent:
    """
    Simple Q-Learning Agent

    - Uses Q-table (dictionary)
    - Epsilon-greedy policy
    - Learns from reward and future estimate
    """

    def __init__(
        self,
        n_actions,
        alpha=0.25,
        gamma=0.9,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.05,
        seed=42,
    ):
        self.n_actions = n_actions

        # Learning parameters
        self.alpha = alpha
        self.gamma = gamma

        # Exploration parameters
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Random generator
        self.rng = np.random.default_rng(seed)

        # Q-table: {(state, action): value}
        self.q_table = {}

    # GET Q VALUE
    def get_q(self, state, action):
        return self.q_table.get((state, action), 0.0)


    # ACTION SELECTION
    def choose_action(self, state):
        # Exploration
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.n_actions))

        # Exploitation
        q_values = [self.get_q(state, a) for a in range(self.n_actions)]
        max_q = max(q_values)

        # Handle ties randomly
        best_actions = [a for a, q in enumerate(q_values) if q == max_q]
        return int(self.rng.choice(best_actions))

    # Q-LEARNING UPDATE
    def update(self, state, action, reward, next_state, done):
        current_q = self.get_q(state, action)

        if done:
            target = reward
        else:
            next_max = max(self.get_q(next_state, a) for a in range(self.n_actions))
            target = reward + self.gamma * next_max

        # Q-learning formula
        new_q = current_q + self.alpha * (target - current_q)
        self.q_table[(state, action)] = new_q

    # DECAY EPSILON
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # GREEDY ACTION (FOR TESTING)
    def best_action(self, state):
        q_values = [self.get_q(state, a) for a in range(self.n_actions)]
        max_q = max(q_values)
        best_actions = [a for a, q in enumerate(q_values) if q == max_q]
        return int(self.rng.choice(best_actions))