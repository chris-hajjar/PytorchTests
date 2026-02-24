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
**Main idea:** Apply the same evolutionary algorithm from CartPole to a classic arcade game. Agent reads the Atari 2600's 128-byte RAM (player position, ghost states, pellet counts, etc.) and outputs one of 9 actions.

**Fitness shaping:** Unlike raw score optimization, fitness now rewards strategic play:
- **Pellet collection**: +5 per pellet (dense progress signal)
- **Ghost hunting**: +20 per ghost eaten (encourages power pellet strategy)
- **Death penalty**: -500 per life lost (strong survival incentive)
- **Level completion**: +200 bonus (ultimate goal)
- **Efficiency**: -0.1 per step (prevents stalling)

This shapes agents to actively collect pellets and complete levels instead of just surviving.

**Takeaway:** Neuroevolution scales from simple control tasks to complex arcade games. Fitness shaping provides dense reward signals that guide evolution toward strategic gameplay. Population diversity (explorers with high mutation noise) prevents local optima.

**Training modes** (all use real-time frame_skip=1):
- **Mini** (30 pop, 128→64→9): ~1-2 hrs / 50 gen — Quick testing
- **Small** (100 pop, 128→64→9): ~4-8 hrs / 100 gen — Small-scale training
- **Stable** (300 pop, 128→128→128→9): ~24-36 hrs / 300 gen — Long stable training
- **Real-time** (300 pop, 128→128→128→9): Runs until Ctrl+C — Production quality

**Logging system:** Each training run creates an organized folder:
```
training_runs/run_20260223_183045_realtime/
  ├── model.pt              # Best model (auto-saved when fitness improves)
  ├── log.txt               # Full console output
  ├── metrics.csv           # Generation stats (best/mean/worst/std)
  ├── config.json           # Training config for reproducibility
  └── checkpoints/          # Periodic checkpoints (gen_0005.pkl, etc.)
```

**Continue training:** You can load any previously saved `.pt` model and seed a new population from it — the saved brain becomes elite #1 and the rest of the population is built by mutating it at varying noise levels for diversity.

**Viewer:** `mspacman_viewer.py` recursively scans the repository for ALL trained models (including those in `training_runs/` folders), displays them sorted by fitness, and lets you watch any agent play in real-time. Shows key info: fitness, generation, network type, and frame skip.

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
