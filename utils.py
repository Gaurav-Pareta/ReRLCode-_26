from dataclasses import dataclass
import numpy as np



# TRAINING CONFIG
@dataclass
class TrainingConfig:
    episodes: int = 3000
    steps_per_episode: int = 20

    # Q-learning parameters
    alpha: float = 0.25
    gamma: float = 0.9

    epsilon: float = 1.0
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.05

    seed: int = 42



# MOVING AVERAGE
def moving_average(values, window=50):
    values = np.array(values, dtype=np.float64)

    if len(values) < window:
        return values

    return np.convolve(values, np.ones(window) / window, mode="valid")


# SIMPLE STATE PRINT
def pretty_state(state, subjects):
    """
    State format:
    (score_bin_math, score_bin_physics, score_bin_chem,
     diff_math, diff_physics, diff_chem)
    """
    n = len(subjects)

    score_bins = state[:n]
    diffs = state[n:]

    return " | ".join(
        f"{subj}: score_bin={sb}, diff={d}"
        for subj, sb, d in zip(subjects, score_bins, diffs)
    )