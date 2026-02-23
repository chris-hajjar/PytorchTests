import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import ale_py
import numpy as np
import os
import sys
import pickle

gym.register_envs(ale_py)


def pick_menu(options):
    """Arrow-key menu selector."""
    import tty, termios
    selected = 0
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        while True:
            sys.stdout.write(f"\r\033[J")
            for i, opt in enumerate(options):
                prefix = "> " if i == selected else "  "
                sys.stdout.write(f"{prefix}{opt}\n")
            sys.stdout.write(f"\033[{len(options)}A")
            sys.stdout.flush()
            ch = sys.stdin.read(1)
            if ch == '\r':
                break
            if ch == '\x1b':
                sys.stdin.read(1)
                arrow = sys.stdin.read(1)
                if arrow == 'A':
                    selected = (selected - 1) % len(options)
                elif arrow == 'B':
                    selected = (selected + 1) % len(options)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        sys.stdout.write(f"\r\033[J")
        sys.stdout.flush()
    return selected


# ============================================================================
# NETWORK ARCHITECTURE (must match training code)
# ============================================================================

class PolicyNetwork(nn.Module):
    """Large network: 128 -> 128 -> 128 -> 9."""
    def __init__(self, state_size=128, action_size=9, hidden_size=128):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def select_action(self, state):
        with torch.no_grad():
            return self.forward(state).argmax().item()


class SmallPolicyNetwork(nn.Module):
    """Small network: 128 -> 64 -> 9."""
    def __init__(self, state_size=128, action_size=9):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def select_action(self, state):
        with torch.no_grad():
            return self.forward(state).argmax().item()


def inject_genome(model, genome):
    """Load genome weights into network."""
    offset = 0
    for param in model.parameters():
        num_weights = param.numel()
        param_weights = genome[offset:offset + num_weights]
        param.data = torch.FloatTensor(param_weights.reshape(param.shape))
        offset += num_weights


def load_genome_from_pt(path):
    """Load a .pt genome export."""
    checkpoint = torch.load(path, weights_only=False)
    config = checkpoint['config']

    if config['network_type'] == 'small':
        model = SmallPolicyNetwork(128, 9)
    else:
        model = PolicyNetwork(128, 9, hidden_size=128)

    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    return model, checkpoint


def load_genome_from_checkpoint(path):
    """Load the best genome from an evolution checkpoint."""
    with open(path, 'rb') as f:
        checkpoint = pickle.load(f)

    population = checkpoint['population']
    best = max(population, key=lambda g: g.fitness)
    config = checkpoint['config']

    if config['network_type'] == 'small':
        model = SmallPolicyNetwork(128, 9)
    else:
        model = PolicyNetwork(128, 9, hidden_size=128)

    inject_genome(model, best.weights)
    model.eval()

    return model, {
        'fitness': best.fitness,
        'generation': best.generation_born,
        'config': config,
        'checkpoint_generation': checkpoint['generation'],
    }


def play_episode(model, device):
    """Play one full episode with rendering. No frame skipping."""
    env = gym.make('ALE/MsPacman-v5', obs_type='ram', render_mode='human')
    observation, _ = env.reset()
    total_reward = 0
    steps = 0

    while True:
        state = torch.FloatTensor(observation.astype(np.float32) / 255.0).unsqueeze(0).to(device)
        action = model.select_action(state)

        observation, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1

        if terminated or truncated:
            break

    env.close()
    return total_reward, steps


def main():
    print("Ms. Pac-Man Genome Viewer\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    script_dir = os.path.dirname(__file__)

    # Find available genomes (only show Ms. Pac-Man ones)
    compatible_pt = []
    for f in sorted(os.listdir(script_dir)):
        if not f.endswith('.pt'):
            continue
        try:
            data = torch.load(os.path.join(script_dir, f), weights_only=False)
            if data.get('config', {}).get('game') == 'mspacman':
                compatible_pt.append(f)
        except Exception:
            pass

    checkpoint_path = os.path.join(script_dir, 'mspacman_neuroevo_checkpoint.pkl')
    has_checkpoint = os.path.exists(checkpoint_path)

    options = []
    option_types = []

    if has_checkpoint:
        options.append("Load best from evolution checkpoint")
        option_types.append('checkpoint')

    for f in compatible_pt:
        options.append(f"Load saved genome: {f}")
        option_types.append(('pt', f))

    options.append("Watch random agent (baseline)")
    option_types.append('random')

    if not options:
        print("No genomes found! Run mspacman_neuroevolution.py first.")
        return

    choice = pick_menu(options)
    selected = option_types[choice]

    if selected == 'checkpoint':
        model, info = load_genome_from_checkpoint(checkpoint_path)
        model = model.to(device)
        print(f"\nLoaded best genome from checkpoint")
        print(f"  Fitness: {info['fitness']:.1f}")
        print(f"  Born generation: {info['generation']}")
        print(f"  Checkpoint at generation: {info['checkpoint_generation']}")
        print(f"  Config: {info['config']['name']}")
        use_model = True

    elif isinstance(selected, tuple) and selected[0] == 'pt':
        path = os.path.join(script_dir, selected[1])
        model, info = load_genome_from_pt(path)
        model = model.to(device)
        print(f"\nLoaded genome from {selected[1]}")
        print(f"  Fitness: {info['fitness']:.1f}")
        print(f"  Generation: {info['generation']}")
        print(f"  Config: {info['config']['name']}")
        use_model = True

    else:
        use_model = False
        print("\nWatching random agent...")

    # Play episodes
    print("\nPress Enter to start an episode, 'q' to quit.\n")

    episode = 0
    while True:
        user_input = input(f"Episode {episode + 1} — press Enter to play (q to quit): ").strip()
        if user_input.lower() == 'q':
            break

        if use_model:
            score, steps = play_episode(model, device)
        else:
            # Random agent
            env = gym.make('ALE/MsPacman-v5', obs_type='ram', render_mode='human')
            env.reset()
            score = 0
            steps = 0
            while True:
                _, reward, terminated, truncated, _ = env.step(env.action_space.sample())
                score += reward
                steps += 1
                if terminated or truncated:
                    break
            env.close()

        episode += 1
        print(f"  Score: {score:.0f}  |  Steps: {steps}")

    print("\nDone!")


if __name__ == '__main__':
    main()
