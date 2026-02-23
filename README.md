# PyTorch ML Experiments

A collection of machine learning experiments built with PyTorch, progressing from basic supervised learning to deep reinforcement learning.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch gymnasium
```

For the Ms. Pac-Man experiments (5), you also need Atari support:

```bash
brew install opencv        # needed to build ale-py from source
pip install ale-py
pip install "autorom[accept-rom-license]"
AutoROM --install-dir "$(python -c 'import ale_py.roms, os; print(os.path.dirname(ale_py.roms.__file__))')" --accept-license
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

---

### 5. Ms. Pac-Man Neuroevolution (`mspacman_neuroevolution.py`)
**Main idea:** Apply the same evolutionary algorithm from CartPole to a classic arcade game. Agent reads the Atari 2600's 128-byte RAM (player position, ghost states, pellet counts, etc.) and outputs one of 9 actions. Fitness = game score minus an idle penalty, so the agent must keep eating pellets rather than hiding in a corner.

**Takeaway:** Neuroevolution scales from simple control tasks to complex arcade games. Same principles (selection + mutation) work with larger observation spaces and longer episodes. An idle penalty in the fitness function prevents degenerate strategies like parking in a safe spot.

**Two modes:**
- Fast (30 population, 128→64→9 network): ~30 min per 100 generations
- Stable (100 population, 128→128→128→9 network): ~2-3 hours per 100 generations

**Viewer:** `mspacman_viewer.py` loads a saved genome or checkpoint and renders the agent playing in real-time (no frame skipping).

## Usage

All scripts live in `experiments/`:

```bash
cd experiments
python basic_regression.py
python basic_RL.py
python cartpole.py
python cartpole_neuroevolution.py
python mspacman_neuroevolution.py
python mspacman_viewer.py
```
