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
                sys.stdout.write(f"{prefix}{opt}\r\n")
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


def play_episode(model, device, frame_skip=4, max_steps=10000):
    """Play one full episode with rendering, matching training frame skip."""
    env = gym.make('ALE/MsPacman-v5', obs_type='ram', render_mode='human')
    observation, _ = env.reset()
    total_reward = 0
    steps = 0

    while steps < max_steps:
        state = torch.FloatTensor(observation.astype(np.float32) / 255.0).unsqueeze(0).to(device)
        action = model.select_action(state)

        # Repeat action for frame_skip frames (matching training)
        for _ in range(frame_skip):
            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            if terminated or truncated or steps >= max_steps:
                break

        if terminated or truncated:
            break

    env.close()
    return total_reward, steps


def find_all_models(base_dir):
    """Recursively find all .pt model files in the repository."""
    models = []

    for root, dirs, files in os.walk(base_dir):
        for f in sorted(files):
            if not f.endswith('.pt'):
                continue

            full_path = os.path.join(root, f)
            rel_path = os.path.relpath(full_path, base_dir)

            try:
                data = torch.load(full_path, weights_only=False)
                if data.get('config', {}).get('game') == 'mspacman':
                    models.append({
                        'filename': f,
                        'rel_path': rel_path,
                        'full_path': full_path,
                        'fitness': data.get('fitness', 0),
                        'generation': data.get('generation', 0),
                        'config_name': data.get('config', {}).get('name', 'unknown'),
                        'network_type': data.get('config', {}).get('network_type', 'unknown'),
                        'frame_skip': data.get('config', {}).get('frame_skip', 4),
                    })
            except Exception as e:
                # Skip files that can't be loaded
                pass

    # Sort by fitness (best first)
    models.sort(key=lambda m: m['fitness'], reverse=True)
    return models


def main():
    print("Ms. Pac-Man Genome Viewer\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    script_dir = os.path.dirname(__file__)

    # Find all models recursively
    print("Scanning for models...")
    models = find_all_models(script_dir)

    checkpoint_path = os.path.join(script_dir, 'mspacman_neuroevo_checkpoint.pkl')
    has_checkpoint = os.path.exists(checkpoint_path)

    options = []
    option_types = []

    if has_checkpoint:
        options.append("Load best from evolution checkpoint")
        option_types.append('checkpoint')

    # Group models by location
    if models:
        # Models in training_runs folders
        training_runs = [m for m in models if 'training_runs' in m['rel_path']]
        root_models = [m for m in models if 'training_runs' not in m['rel_path']]

        print(f"Found {len(models)} model(s):")
        if training_runs:
            print(f"  - {len(training_runs)} in training_runs/")
        if root_models:
            print(f"  - {len(root_models)} saved model(s)")
        print()

        if training_runs:
            options.append("─── Training Run Models ───")
            option_types.append('separator')
            for m in training_runs:
                # Extract run name from path
                parts = m['rel_path'].split(os.sep)
                if len(parts) >= 2 and parts[0] == 'training_runs':
                    run_name = parts[1]
                    display = f"  {run_name} | fitness: {m['fitness']:.0f} | gen: {m['generation']} | fs={m['frame_skip']}"
                else:
                    display = f"  {m['rel_path']} | fitness: {m['fitness']:.0f}"
                options.append(display)
                option_types.append(('model', m))

        if root_models:
            if training_runs:  # Add separator if we already have training runs
                options.append("─── Saved Models ───")
                option_types.append('separator')
            for m in root_models:
                display = f"  {m['filename']} | fitness: {m['fitness']:.0f} | gen: {m['generation']} | {m['network_type']} | fs={m['frame_skip']}"
                options.append(display)
                option_types.append(('model', m))

    options.append("─── Baseline ───")
    option_types.append('separator')
    options.append("Watch random agent")
    option_types.append('random')

    if len(options) == 3:  # Only baseline option
        print("No trained models found! Run mspacman_neuroevolution.py first.")
        return

    choice = pick_menu(options)
    selected = option_types[choice]

    # Skip separators
    while selected == 'separator':
        choice = (choice + 1) % len(options)
        selected = option_types[choice]

    if selected == 'checkpoint':
        model, info = load_genome_from_checkpoint(checkpoint_path)
        model = model.to(device)
        print(f"\nLoaded best genome from checkpoint")
        print(f"  Fitness: {info['fitness']:.1f}")
        print(f"  Born generation: {info['generation']}")
        print(f"  Checkpoint at generation: {info['checkpoint_generation']}")
        print(f"  Config: {info['config']['name']}")
        config = info['config']
        use_model = True

    elif isinstance(selected, tuple) and selected[0] == 'model':
        model_info = selected[1]
        model, info = load_genome_from_pt(model_info['full_path'])
        model = model.to(device)
        print(f"\nLoaded: {model_info['rel_path']}")
        print(f"  Fitness: {info['fitness']:.1f}")
        print(f"  Generation: {info['generation']}")
        print(f"  Config: {info['config']['name']}")
        print(f"  Network: {info['config']['network_type']}")
        print(f"  Frame skip: {info['config'].get('frame_skip', 4)}")
        config = info['config']
        use_model = True

    else:
        use_model = False
        config = {'frame_skip': 1, 'max_steps': 10000}  # Default to frame_skip=1 for random
        print("\nWatching random agent...")

    # Play episodes
    print("\nPress Enter to start an episode, 'q' to quit.\n")

    episode = 0
    while True:
        user_input = input(f"Episode {episode + 1} — press Enter to play (q to quit): ").strip()
        if user_input.lower() == 'q':
            break

        frame_skip = config.get('frame_skip', 4)
        max_steps = config.get('max_steps', 10000)

        if use_model:
            score, steps = play_episode(model, device, frame_skip, max_steps)
        else:
            # Random agent with frame skip
            env = gym.make('ALE/MsPacman-v5', obs_type='ram', render_mode='human')
            env.reset()
            score = 0
            steps = 0
            while steps < max_steps:
                action = env.action_space.sample()
                for _ in range(frame_skip):
                    _, reward, terminated, truncated, _ = env.step(action)
                    score += reward
                    steps += 1
                    if terminated or truncated or steps >= max_steps:
                        break
                if terminated or truncated:
                    break
            env.close()

        episode += 1
        print(f"  Score: {score:.0f}  |  Steps: {steps}")

    print("\nDone!")


if __name__ == '__main__':
    main()
