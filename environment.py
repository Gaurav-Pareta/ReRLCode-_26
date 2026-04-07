import numpy as np


class StudyEnvironment:
    """
    Simplified RL Environment for Study Planning

    - Subjects: Math, Physics, Chemistry
    - State: (score_bins, difficulties)
    - Action: choose subject to study (0,1,2)
    - Episode: multiple study steps (e.g., 20)
    """

    SUBJECTS = ["Math", "Physics", "Chemistry"]

    def __init__(self, steps_per_episode=20, score_bin_size=10, seed=42):
        self.steps_per_episode = steps_per_episode
        self.score_bin_size = score_bin_size
        self.rng = np.random.default_rng(seed)

    def reset(self):
        """
        Initialize new episode
        """
        self.step_count = 0

        # Start scores in mid range
        self.scores = self.rng.integers(40, 70, size=3)

        # 0 = easy, 1 = medium, 2 = hard
        self.difficulties = self.rng.integers(0, 3, size=3)

        self.last_studied = [0, 0, 0]

        return self._get_state()

    def step(self, action):
        """
        Perform one study action
        """
        action = int(action)
    
        prev_score = int(self.scores[action])
        difficulty = int(self.difficulties[action])
    
        min_before = min(self.scores)
    
        # Get score change
        delta = self._score_change(difficulty)
    
        # Update score
        new_score = int(np.clip(prev_score + delta, 0, 100))
        self.scores[action] = new_score
    
        # Update last studied counters
        self.last_studied = [x + 1 for x in self.last_studied]
        self.last_studied[action] = 0
    
        # Reward based on direct improvement (delta)
        reward = float(delta)
    
        # Penalty for regressions (stronger)
        if delta < 0:
            reward = delta * 1.5
    
        # Hard-subject bonus (only if progress)
        if difficulty == 2 and delta > 0:
            reward += 2.0
    
        # Extra catch-up bonus for low-scoring subject
        if prev_score < 50 and delta > 0:
            reward += 0.5
    
        # Weakest subject improvement bonus (global balancing)
        min_after = min(self.scores)
        reward += (min_after - min_before) * 2.0
    
        # Small anti-stale penalty for repeating identical subject
        if hasattr(self, "last_action") and self.last_action == action:
            reward -= 0.5
        self.last_action = action
    
        self.step_count += 1
        done = self.step_count >= self.steps_per_episode
    
        info = {
            "subject": self.SUBJECTS[action],
            "delta": int(new_score - prev_score),
            "scores": self.scores.copy(),
        }
    
        return self._get_state(), reward, done, info
    def _get_state(self):
        """
        State = discretized scores + difficulties
        """
        score_bins = tuple(int(s // self.score_bin_size) for s in self.scores)
        difficulties = tuple(int(d) for d in self.difficulties)

        return score_bins + difficulties

    def _score_change(self, difficulty):
        if difficulty == 0:  # easy
            return int(self.rng.integers(-1, 6))   
        elif difficulty == 1:  # medium
            return int(self.rng.integers(-2, 5))   
        else:  # hard
            return int(self.rng.integers(-2, 4))  