# PyTorch ML Experiments

A collection of machine learning experiments built with PyTorch, progressing from basic supervised learning to deep reinforcement learning.

## Experiments

### 1. Linear Regression (`venv/basic_regression.py`)
A minimal supervised learning example. Trains a single `nn.Linear` layer to recover the relationship `y = 2x + 1` from noisy synthetic data using MSE loss and SGD.

- **Concepts:** forward pass, backpropagation, gradient descent
- **Key takeaway:** PyTorch autograd handles all the calculus — you just define the model and loss.

### 2. Multi-Armed Bandit (`venv/basic_RL.py`)
Intro to reinforcement learning. A small neural network learns which of 3 slot machines has the best payout using epsilon-greedy exploration and Q-value estimation.

- **Concepts:** exploration vs exploitation, Q-learning, epsilon-greedy policy
- **Key takeaway:** Even a simple linear model can learn optimal action selection from reward signals alone.

### 3. CartPole DQN (`venv/cartpole.py`)
A full Deep Q-Network (DQN) agent that learns to balance a pole on a cart using the Gymnasium `CartPole-v1` environment. Includes experience replay, a target network, epsilon decay, checkpoint saving/resuming, and a random-agent baseline comparison.

- **Concepts:** deep Q-learning, replay buffer, target network, epsilon decay, model checkpointing
- **Key takeaway:** Combining a neural network with RL fundamentals (replay, target nets) is enough to solve classic control tasks.

## Setup

```bash
cd venv
python -m venv .
source bin/activate
pip install torch gymnasium
```

## Usage

```bash
# Run from the venv directory
python basic_regression.py
python basic_RL.py
python cartpole.py
```
