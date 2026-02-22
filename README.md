# PyTorch ML Experiments

A collection of machine learning experiments built with PyTorch, progressing from basic supervised learning to deep reinforcement learning.

## Setup

```bash
cd venv
python -m venv .
source bin/activate
pip install torch gymnasium
```

## Experiments

### 1. Linear Regression (`basic_regression.py`)
**Main idea:** Train a single linear layer to recover `y = 2x + 1` from noisy data using gradient descent.

**Takeaway:** PyTorch autograd handles all the calculus — you just define the model and loss.

---

### 2. Multi-Armed Bandit (`basic_RL.py`)
**Main idea:** A neural network learns which of 3 slot machines has the best payout using epsilon-greedy exploration.

**Takeaway:** Even a simple model can learn optimal action selection from reward signals alone (no labels needed).

---

### 3. CartPole DQN (`cartpole.py`)
**Main idea:** Deep Q-Network learns to balance a pole on a cart using experience replay, target networks, and epsilon decay.

**Takeaway:** Combining neural networks with RL fundamentals (replay buffer, target nets) solves classic control tasks through gradient-based learning.

---

### 4. CartPole Neuroevolution (`cartpole_neuroevolution.py`)
**Main idea:** Solve CartPole using evolution instead of gradients - treat network weights as genes, evaluate fitness, select winners, mutate, repeat.

**Takeaway:** Evolution is a viable alternative to backpropagation. No replay buffer, no target network, no optimizer - just selection and mutation. Simpler conceptually but requires more environment interactions.

**Two modes:**
- Fast (50 population, small network): ~10 min
- Stable (150 population, full network): ~60 min

## Usage

```bash
python basic_regression.py
python basic_RL.py
python cartpole.py
python cartpole_neuroevolution.py
```
