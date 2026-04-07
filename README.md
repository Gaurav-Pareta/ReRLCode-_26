# Smart Study Time Allocator (Q-learning, clean rebuild)

## Overview
This project implements a Reinforcement Learning (RL) agent using Q-learning to optimize study time allocation across three subjects: Math, Physics, and Chemistry. The goal is to maximize weekly score improvement by learning which subject to study each day.

## Project Structure
- `main.py`: Main script that runs the training, evaluation, and simulation
- `agent.py`: Q-learning agent implementation
- `environment.py`: Study environment simulation
- `utils.py`: Utility functions and configuration
- `outputs/`: Directory for saved outputs (Q-table, plots)

## Key Components

### Environment (StudyEnvironment)
The environment simulates a student's study session over multiple days (episodes).

**Subjects**: Math (0), Physics (1), Chemistry (2)

**State Representation**:
- `score_bins`: Current scores binned into 10-point ranges (0-10)
- `difficulties`: Subject difficulty levels (0=easy, 1=medium, 2=hard)

State is a tuple: `(score_bin_math, score_bin_physics, score_bin_chemistry, diff_math, diff_physics, diff_chemistry)`

**Actions**: Choose which subject to study (0, 1, or 2)

**Score Improvement Model**:
- Easy subjects: score change ∈ [-1, 5]
- Medium subjects: score change ∈ [-2, 4]
- Hard subjects: score change ∈ [-2, 3]

**Episode Length**: 20 study days per episode

### Agent (QLearningAgent)
Implements Q-learning algorithm with epsilon-greedy exploration.

**Parameters**:
- `alpha` (learning rate): 0.25
- `gamma` (discount factor): 0.9
- `epsilon` (exploration rate): starts at 1.0, decays to 0.05
- `epsilon_decay`: 0.995

**Q-table**: Dictionary storing state-action values

### Reward Function
The reward encourages balanced improvement while providing bonuses for challenging subjects:

Base reward based on score delta:
- +10 if delta ≥ 5
- +5 if delta ∈ [1, 4]
- -5 if delta = 0
- -10 if delta < 0

Additional shaping:
- Hard subject bonus: +2 if improved hard subject
- Low-score catch-up bonus: +0.5 if improved subject below 50
- Global balancing: +2×(min_score_improvement) to encourage weakest subject improvement
- Anti-stale penalty: -0.5 for repeating the same subject consecutively

## Training Process
1. Initialize environment and agent
2. Train for 5000 episodes
3. Each episode: 20 study decisions
4. Agent learns from rewards and state transitions
5. Epsilon decays over time for more exploitation

## Evaluation
After training:
- Performance metrics (average reward, stability, improvement)
- Score trend visualization across training
- Action distribution analysis
- Test week simulation with greedy policy

## Outputs
- `outputs/q_table.pkl`: Trained Q-table
- `outputs/rewards.png`: Training progress plot
- Console output: Training progress, metrics, simulation results

## Dependencies
- numpy
- matplotlib

## How to Run
1. Install dependencies:
   ```bash
   pip install numpy matplotlib
   ```

2. Run the main script:
   ```bash
   python main.py
   ```

## Results Interpretation
The agent learns to:
- Prioritize subjects that yield the most improvement
- Balance study time across subjects
- Avoid over-studying easy subjects
- Focus on difficult or neglected subjects when beneficial

Training shows convergence in reward and balanced score improvement across all subjects.

