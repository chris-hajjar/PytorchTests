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
**Main idea:** Apply the same evolutionary algorithm from CartPole to a classic arcade game. Agent reads the Atari 2600's 128-byte RAM (player position, ghost states, pellet counts, etc.) and outputs one of 9 actions. Fitness = raw game score averaged over multiple episodes.

**Takeaway:** Neuroevolution scales from simple control tasks to complex arcade games. Same principles (selection + mutation) work with larger observation spaces and longer episodes. Population diversity (explorers with high mutation noise) prevents degenerate strategies.

**Training profiles:**
- Quick test (30 pop, 128→64→9 network): ~15-20 min / 50 gen
- Fast iteration (100 pop, 128→64→9 network): ~1-2 hrs / 100 gen
- Stable learning (300 pop, 128→128→128→9 network): ~6-8 hrs / 300 gen
- Overnight (300 pop, 128→128→128→9 network): runs until Ctrl+C, auto-saves best genome

**Continue training:** You can pick a previously saved `.pt` genome and seed a new population from it — the saved brain becomes elite #1 and the rest of the population is built by mutating it at varying noise levels.

**Viewer:** `mspacman_viewer.py` loads a saved `.pt` genome and renders the agent playing in real-time.

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
