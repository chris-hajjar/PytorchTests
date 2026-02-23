import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import ale_py
import numpy as np
import pickle
import os
import sys
import signal
from datetime import datetime

# Register ALE environments with gymnasium
gym.register_envs(ale_py)

# Checkpoint path
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), 'mspacman_neuroevo_checkpoint.pkl')


def pick_menu(options):
    """Arrow-key menu selector."""
    import tty, termios
    selected = 0
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        while True:
            # Clear and draw menu
            sys.stdout.write(f"\r\033[J")
            for i, opt in enumerate(options):
                prefix = "> " if i == selected else "  "
                sys.stdout.write(f"{prefix}{opt}\r\n")
            sys.stdout.write(f"\033[{len(options)}A")  # move cursor back up
            sys.stdout.flush()
            # Read keypress
            ch = sys.stdin.read(1)
            if ch == '\r':
                break
            if ch == '\x1b':
                sys.stdin.read(1)  # skip [
                arrow = sys.stdin.read(1)
                if arrow == 'A':
                    selected = (selected - 1) % len(options)
                elif arrow == 'B':
                    selected = (selected + 1) % len(options)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        sys.stdout.write(f"\r\033[J")  # clean up
        sys.stdout.flush()
    return selected


# ============================================================================
# GENOME REPRESENTATION
# ============================================================================

class Genome:
    """Represents a single neural network as a genome."""
    def __init__(self, weights, fitness=0.0, generation_born=0):
        self.weights = weights  # Flattened numpy array of all network parameters
        self.fitness = fitness
        self.generation_born = generation_born


# ============================================================================
# NETWORK ARCHITECTURE
# ============================================================================

class PolicyNetwork(nn.Module):
    """Large network for Ms. Pac-Man: 128 → 128 → 128 → 9."""
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
        """Deterministic policy: argmax over action values."""
        with torch.no_grad():
            q_values = self.forward(state)
            return q_values.argmax().item()


class SmallPolicyNetwork(nn.Module):
    """Smaller network for fast iteration: 128 → 64 → 9."""
    def __init__(self, state_size=128, action_size=9):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def select_action(self, state):
        """Deterministic policy: argmax over action values."""
        with torch.no_grad():
            q_values = self.forward(state)
            return q_values.argmax().item()


# ============================================================================
# GENOME <-> NETWORK CONVERSION
# ============================================================================

def extract_genome(model):
    """Extract all network weights as a flat numpy array."""
    weights = []
    for param in model.parameters():
        weights.append(param.data.cpu().numpy().flatten())
    return np.concatenate(weights)


def inject_genome(model, genome):
    """Load genome weights back into network."""
    offset = 0
    for param in model.parameters():
        num_weights = param.numel()
        param_weights = genome[offset:offset + num_weights]
        param.data = torch.FloatTensor(param_weights.reshape(param.shape))
        offset += num_weights


# ============================================================================
# GENOME EVALUATION
# ============================================================================

def evaluate_genome(genome, env, model, device, k_episodes=1, max_steps=5000, frame_skip=4):
    """
    Evaluate a genome by running k episodes and averaging raw game score.

    No fitness shaping — just the game score. Degenerate strategies are
    handled by population diversity, not artificial penalties.

    Args:
        genome: Genome object to evaluate
        env: Gymnasium environment
        model: PyTorch model to inject weights into
        device: torch device
        k_episodes: Number of episodes to average over
        max_steps: Maximum frames per episode (prevents infinite loops)
        frame_skip: Repeat each action for this many frames (speeds up evaluation)

    Returns:
        Average fitness (total reward) across k episodes
    """
    inject_genome(model, genome.weights)
    model.eval()

    total_rewards = []

    for _ in range(k_episodes):
        observation, _ = env.reset()
        episode_reward = 0
        steps = 0

        while steps < max_steps:
            # Convert RAM bytes to float tensor (normalize 0-255 to 0-1)
            state = torch.FloatTensor(observation.astype(np.float32) / 255.0).unsqueeze(0).to(device)
            action = model.select_action(state)

            # Frame skipping: repeat action for frame_skip frames
            frame_reward = 0
            for _ in range(frame_skip):
                observation, reward, terminated, truncated, _ = env.step(action)
                frame_reward += reward
                steps += 1
                if terminated or truncated or steps >= max_steps:
                    break

            episode_reward += frame_reward

            if terminated or truncated:
                break

        total_rewards.append(episode_reward)

    return np.mean(total_rewards)


# ============================================================================
# EVOLUTIONARY OPERATORS
# ============================================================================

def select_parents(population, elite_count, truncation_ratio):
    """
    Select top performers for breeding.

    Returns:
        elites, breeding_pool (both lists of Genome objects)
    """
    sorted_pop = sorted(population, key=lambda g: g.fitness, reverse=True)
    elites = sorted_pop[:elite_count]
    breeding_pool_size = max(elite_count, int(len(population) * truncation_ratio))
    breeding_pool = sorted_pop[:breeding_pool_size]
    return elites, breeding_pool


def mutate_genome(parent, mutation_std, generation):
    """Create offspring by adding Gaussian noise to parent weights."""
    noise = np.random.normal(0, mutation_std, size=parent.weights.shape)
    child_weights = parent.weights + noise
    return Genome(child_weights, fitness=0.0, generation_born=generation)


def generate_offspring(breeding_pool, target_size, elite_count, mutation_std, generation, explorer_count=0, explorer_std_mult=3.0):
    """Generate next generation from breeding pool, plus explorers for diversity.

    Explorers are mutated from the current best genome with high noise —
    they inherit enough structure to not be useless but explore further out.
    """
    offspring = []
    num_offspring_needed = target_size - elite_count - explorer_count

    for _ in range(num_offspring_needed):
        parent = np.random.choice(breeding_pool)
        child = mutate_genome(parent, mutation_std, generation)
        offspring.append(child)

    # Explorers: heavy mutations of the best genome
    best = max(breeding_pool, key=lambda g: g.fitness)
    for _ in range(explorer_count):
        child = mutate_genome(best, mutation_std * explorer_std_mult, generation)
        offspring.append(child)

    return offspring


# ============================================================================
# CHECKPOINT SYSTEM
# ============================================================================

def save_checkpoint(population, generation, history, config, path):
    """Save evolution state to disk."""
    checkpoint = {
        'population': population,
        'generation': generation,
        'history': history,
        'config': config,
        'timestamp': datetime.now().isoformat()
    }
    with open(path, 'wb') as f:
        pickle.dump(checkpoint, f)


def load_checkpoint(path):
    """Load evolution state from disk."""
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_genome_from_pt(path):
    """Load a saved .pt file and return a Genome object + config."""
    checkpoint = torch.load(path, weights_only=False)
    config = checkpoint['config']

    # Reconstruct model to extract flat weights
    if config['network_type'] == 'small':
        model = SmallPolicyNetwork(128, 9)
    else:
        model = PolicyNetwork(128, 9, hidden_size=128)

    model.load_state_dict(checkpoint['state_dict'])
    weights = extract_genome(model)

    genome = Genome(
        weights,
        fitness=checkpoint.get('fitness', 0.0),
        generation_born=checkpoint.get('generation', 0),
    )
    return genome, config


def seed_population_from_genome(genome, population_size, mutation_std):
    """Build a new population seeded from a single known-good genome.

    The seed genome becomes elite #1. The rest of the population is
    generated by mutating it at varying noise levels for diversity:
      - 20% close mutations (0.5x std) — fine-tune neighborhood
      - 60% normal mutations (1x std) — standard exploration
      - 20% far mutations (3x std) — wide exploration
    """
    population = [Genome(genome.weights.copy(), fitness=genome.fitness, generation_born=0)]

    remaining = population_size - 1
    n_close = int(remaining * 0.2)
    n_far = int(remaining * 0.2)
    n_normal = remaining - n_close - n_far

    for _ in range(n_close):
        noise = np.random.normal(0, mutation_std * 0.5, size=genome.weights.shape)
        population.append(Genome(genome.weights + noise, generation_born=0))

    for _ in range(n_normal):
        noise = np.random.normal(0, mutation_std, size=genome.weights.shape)
        population.append(Genome(genome.weights + noise, generation_born=0))

    for _ in range(n_far):
        noise = np.random.normal(0, mutation_std * 3.0, size=genome.weights.shape)
        population.append(Genome(genome.weights + noise, generation_born=0))

    return population


def export_best_to_pytorch(genome, config, save_path):
    """Export best genome as loadable PyTorch checkpoint."""
    state_size = 128
    action_size = 9

    if config['network_type'] == 'small':
        model = SmallPolicyNetwork(state_size, action_size)
    else:
        model = PolicyNetwork(state_size, action_size, hidden_size=128)

    inject_genome(model, genome.weights)

    torch.save({
        'state_dict': model.state_dict(),
        'fitness': genome.fitness,
        'generation': genome.generation_born,
        'config': config,
    }, save_path)

    print(f"Best genome exported to: {save_path}")


# ============================================================================
# MAIN EVOLUTION LOOP
# ============================================================================

def evolve(config, resume_from=None, seed_genome=None):
    """
    Main neuroevolution loop for Ms. Pac-Man.

    Args:
        config: Dictionary with hyperparameters
        resume_from: Path to checkpoint file (if resuming)
        seed_genome: Genome object to seed population from (if continuing from .pt)

    Returns:
        Best Genome object, history dict, stop_requested flag
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make('ALE/MsPacman-v5', obs_type='ram')

    # Graceful Ctrl+C shutdown — stops immediately after current genome
    stop_requested = False
    prev_handler = signal.getsignal(signal.SIGINT)

    def handle_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
        print("\n\n  Stop requested — saving and exiting...")

    signal.signal(signal.SIGINT, handle_stop)

    state_size = 128
    action_size = 9

    if config['network_type'] == 'small':
        model = SmallPolicyNetwork(state_size, action_size).to(device)
    else:
        model = PolicyNetwork(state_size, action_size, hidden_size=128).to(device)

    # Initialize or resume
    if resume_from and os.path.exists(resume_from):
        checkpoint = load_checkpoint(resume_from)
        population = checkpoint['population']
        start_gen = checkpoint['generation'] + 1
        history = checkpoint['history']
        print(f"Resumed from generation {checkpoint['generation']}")
    elif seed_genome is not None:
        print(f"Seeding population of {config['population_size']} from saved genome (fitness: {seed_genome.fitness:.1f})...")
        genome_size = len(seed_genome.weights)
        print(f"Genome size: {genome_size} parameters")
        population = seed_population_from_genome(
            seed_genome, config['population_size'], config['mutation_std']
        )
        start_gen = 0
        history = {
            'best_fitness': [],
            'mean_fitness': [],
            'worst_fitness': [],
        }
    else:
        print(f"Initializing random population of {config['population_size']}...")
        genome_size = sum(p.numel() for p in model.parameters())
        print(f"Genome size: {genome_size} parameters")

        population = []
        for i in range(config['population_size']):
            random_weights = extract_genome(model)  # Get shape template
            random_weights = np.random.randn(len(random_weights)) * 0.1
            population.append(Genome(random_weights, fitness=0.0, generation_born=0))

        start_gen = 0
        history = {
            'best_fitness': [],
            'mean_fitness': [],
            'worst_fitness': [],
        }

    # Evolution loop — if resuming past max_generations, run 50 more
    if start_gen >= config['max_generations']:
        config['max_generations'] = start_gen + 50
        print(f"Extending evolution to generation {config['max_generations']}")

    for generation in range(start_gen, config['max_generations']):
        print(f"\n{'='*60}")
        print(f"Generation {generation}")
        print(f"{'='*60}")

        # Evaluate population
        fitnesses = []
        for i, genome in enumerate(population):
            if stop_requested:
                break

            fitness = evaluate_genome(
                genome, env, model, device,
                k_episodes=config['fitness_episodes'],
                max_steps=config['max_steps'],
                frame_skip=config['frame_skip']
            )
            genome.fitness = fitness
            fitnesses.append(fitness)

            if (i + 1) % 5 == 0 or (i + 1) == len(population):
                print(f"  Evaluated: {i+1}/{len(population)} genomes", end='\r')

        print()  # Newline after progress

        # Stop immediately — save checkpoint and break before selection
        if stop_requested:
            print(f"\n  Stopping at generation {generation}")
            save_checkpoint(population, generation, history, config, CHECKPOINT_PATH)
            print(f"  Checkpoint saved at generation {generation}")
            break

        # Track statistics
        best_fitness = max(fitnesses)
        mean_fitness = np.mean(fitnesses)
        worst_fitness = min(fitnesses)

        history['best_fitness'].append(best_fitness)
        history['mean_fitness'].append(mean_fitness)
        history['worst_fitness'].append(worst_fitness)

        # Print stats
        print(f"  Best:  {best_fitness:7.1f}  |  Mean:  {mean_fitness:7.1f}  |  Worst:  {worst_fitness:7.1f}")

        # Selection
        elites, breeding_pool = select_parents(
            population,
            config['elite_count'],
            config['truncation_ratio']
        )

        print(f"  Elites: {config['elite_count']}  |  Breeding pool: {len(breeding_pool)}")

        # Generate offspring (with explorers for diversity)
        offspring = generate_offspring(
            breeding_pool,
            config['population_size'],
            config['elite_count'],
            config['mutation_std'],
            generation + 1,
            explorer_count=config.get('explorer_count', 0),
            explorer_std_mult=config.get('explorer_std_mult', 3.0)
        )

        # Create next generation (elites + offspring + explorers)
        population = elites + offspring

        # Save checkpoint every N generations
        if generation % config['checkpoint_interval'] == 0:
            save_checkpoint(population, generation, history, config, CHECKPOINT_PATH)
            print(f"  Checkpoint saved at generation {generation}")

    # Restore previous signal handler
    signal.signal(signal.SIGINT, prev_handler)

    # Final checkpoint (if we didn't already save on stop)
    if not stop_requested:
        save_checkpoint(population, generation, history, config, CHECKPOINT_PATH)

    # Best genome = top elite from the final generation (consistent performer, not a one-time outlier)
    best_genome = max(population, key=lambda g: g.fitness)

    env.close()
    return best_genome, history, stop_requested


# ============================================================================
# TESTING
# ============================================================================

def test_best_genome(genome, config, num_episodes=5):
    """Test best genome without rendering."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make('ALE/MsPacman-v5', obs_type='ram')

    state_size = 128
    action_size = 9

    if config['network_type'] == 'small':
        model = SmallPolicyNetwork(state_size, action_size).to(device)
    else:
        model = PolicyNetwork(state_size, action_size, hidden_size=128).to(device)

    inject_genome(model, genome.weights)
    model.eval()

    print(f"\n{'='*60}")
    print(f"Testing best genome (Fitness: {genome.fitness:.1f}, Gen: {genome.generation_born})")
    print(f"{'='*60}")

    rewards = []
    for episode in range(num_episodes):
        observation, _ = env.reset()
        total_reward = 0
        steps = 0

        while steps < config.get('max_steps', 10000):
            state = torch.FloatTensor(observation.astype(np.float32) / 255.0).unsqueeze(0).to(device)
            action = model.select_action(state)

            for _ in range(config.get('frame_skip', 4)):
                observation, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                steps += 1
                if terminated or truncated or steps >= config.get('max_steps', 10000):
                    break

            if terminated or truncated:
                break

        rewards.append(total_reward)
        print(f"  Episode {episode+1}: {total_reward:.0f}")

    print(f"\nAverage: {np.mean(rewards):.1f} +/- {np.std(rewards):.1f}")

    env.close()
    return rewards


def test_random_baseline(num_episodes=5, max_steps=5000, frame_skip=4):
    """Test random agent for comparison."""
    env = gym.make('ALE/MsPacman-v5', obs_type='ram')

    print(f"\n{'='*60}")
    print(f"Random baseline agent")
    print(f"{'='*60}")

    rewards = []
    for episode in range(num_episodes):
        env.reset()
        total_reward = 0
        steps = 0

        while steps < max_steps:
            action = env.action_space.sample()

            for _ in range(frame_skip):
                _, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                steps += 1
                if terminated or truncated or steps >= max_steps:
                    break

            if terminated or truncated:
                break

        rewards.append(total_reward)
        print(f"  Episode {episode+1}: {total_reward:.0f}")

    print(f"\nAverage: {np.mean(rewards):.1f} +/- {np.std(rewards):.1f}")

    env.close()
    return rewards


# ============================================================================
# CONFIGURATION PROFILES
# ============================================================================

QUICK_CONFIG = {
    'game': 'mspacman',
    'name': 'Quick test profile',
    'population_size': 30,
    'network_type': 'small',       # 128 -> 64 -> 9
    'fitness_episodes': 1,
    'mutation_std': 0.1,
    'elite_count': 2,
    'truncation_ratio': 0.2,
    'explorer_count': 2,
    'explorer_std_mult': 3.0,
    'max_generations': 50,
    'checkpoint_interval': 5,
    'max_steps': 3000,
    'frame_skip': 4,
}

FAST_CONFIG = {
    'game': 'mspacman',
    'name': 'Fast iteration profile',
    'population_size': 100,
    'network_type': 'small',       # 128 -> 64 -> 9
    'fitness_episodes': 1,
    'mutation_std': 0.1,
    'elite_count': 3,
    'truncation_ratio': 0.2,
    'explorer_count': 5,            # heavy-mutated scouts per generation
    'explorer_std_mult': 3.0,       # explorers get 3x normal mutation noise
    'max_generations': 100,
    'checkpoint_interval': 5,
    'max_steps': 5000,
    'frame_skip': 4,
}

STABLE_CONFIG = {
    'game': 'mspacman',
    'name': 'Stable learning profile',
    'population_size': 300,
    'network_type': 'large',       # 128 -> 128 -> 128 -> 9
    'fitness_episodes': 3,
    'mutation_std': 0.05,
    'elite_count': 10,
    'truncation_ratio': 0.15,
    'explorer_count': 15,           # heavy-mutated scouts per generation
    'explorer_std_mult': 3.0,       # explorers get 3x normal mutation noise
    'max_generations': 300,
    'checkpoint_interval': 10,
    'max_steps': 10000,
    'frame_skip': 4,
}

OVERNIGHT_CONFIG = {
    'game': 'mspacman',
    'name': 'Overnight profile',
    'population_size': 300,
    'network_type': 'large',       # 128 -> 128 -> 128 -> 9
    'fitness_episodes': 3,
    'mutation_std': 0.05,
    'elite_count': 10,
    'truncation_ratio': 0.15,
    'explorer_count': 15,
    'explorer_std_mult': 3.0,
    'max_generations': 999999,     # effectively infinite — stop with Ctrl+C
    'checkpoint_interval': 5,      # save often since we don't know when it stops
    'max_steps': 10000,
    'frame_skip': 4,
}


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    print("Ms. Pac-Man Neuroevolution\n")

    # Scan for saved .pt genomes
    experiments_dir = os.path.dirname(__file__)
    pt_files = sorted([f for f in os.listdir(experiments_dir) if f.endswith('.pt')])

    # Menu
    options = [
        f"New run - {QUICK_CONFIG['name']} (30 pop, ~15-20 min / 50 gen)",
        f"New run - {FAST_CONFIG['name']} (100 pop, ~1-2 hrs / 100 gen)",
        f"New run - {STABLE_CONFIG['name']} (300 pop, ~6-8 hrs / 300 gen)",
        f"New run - {OVERNIGHT_CONFIG['name']} (300 pop, runs until Ctrl+C)",
    ]

    if pt_files:
        options.append("Continue training (from saved .pt genome)")

    if os.path.exists(CHECKPOINT_PATH):
        options.append("Resume training (from checkpoint)")

    choice = pick_menu(options)

    # Determine config, resume status, and seed genome
    seed_genome = None

    if choice == 0:
        config = QUICK_CONFIG
        resume = False
    elif choice == 1:
        config = FAST_CONFIG
        resume = False
    elif choice == 2:
        config = STABLE_CONFIG
        resume = False
    elif choice == 3:
        config = OVERNIGHT_CONFIG
        resume = False
    elif pt_files and choice == 4:
        # Pick which .pt file to load
        print("\nSelect saved genome:\n")
        pt_choice = pick_menu(pt_files)
        pt_path = os.path.join(experiments_dir, pt_files[pt_choice])
        seed_genome, saved_config = load_genome_from_pt(pt_path)
        network_type = saved_config['network_type']

        print(f"\nLoaded: {pt_files[pt_choice]}")
        print(f"  Network: {network_type}  |  Fitness: {seed_genome.fitness:.1f}  |  Gen: {seed_genome.generation_born}")

        # Pick training profile (only show matching network types)
        all_profiles = [
            ('Quick test', QUICK_CONFIG),
            ('Fast iteration', FAST_CONFIG),
            ('Stable learning', STABLE_CONFIG),
            ('Overnight', OVERNIGHT_CONFIG),
        ]
        matching = [(name, cfg) for name, cfg in all_profiles if cfg['network_type'] == network_type]

        if not matching:
            print(f"\nNo profiles match network type '{network_type}'!")
            return

        print(f"\nSelect training profile:\n")
        profile_labels = [f"{name} ({cfg['population_size']} pop, {cfg['max_generations']} gen)" for name, cfg in matching]
        profile_choice = pick_menu(profile_labels)
        config = dict(matching[profile_choice][1])  # copy so we don't mutate the original
        resume = False
    else:
        checkpoint = load_checkpoint(CHECKPOINT_PATH)
        config = checkpoint['config']
        resume = True

    # Print configuration
    print(f"\nConfiguration: {config['name']}")
    print(f"  Population: {config['population_size']}")
    print(f"  Network: {config['network_type']}")
    print(f"  Fitness episodes: {config['fitness_episodes']}")
    print(f"  Mutation std: {config['mutation_std']}")
    print(f"  Elite count: {config['elite_count']}")
    print(f"  Max steps/episode: {config['max_steps']}")
    print(f"  Frame skip: {config['frame_skip']}")
    print()

    # Run evolution
    best_genome, history, was_stopped = evolve(
        config,
        resume_from=CHECKPOINT_PATH if resume else None,
        seed_genome=seed_genome,
    )

    # Print evolution summary
    print(f"\n{'='*60}")
    print(f"Evolution Summary")
    print(f"{'='*60}")
    print(f"Best fitness: {best_genome.fitness:.1f}")
    print(f"Best generation: {best_genome.generation_born}")
    print(f"Total generations: {len(history['best_fitness'])}")
    print(f"Final mean fitness: {history['mean_fitness'][-1]:.1f}")

    # Auto-export on Ctrl+C stop (so there's always a saved model)
    if was_stopped:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        auto_path = os.path.join(os.path.dirname(__file__), f'overnight_best_{timestamp}.pt')
        export_best_to_pytorch(best_genome, config, auto_path)

    # Test best genome
    test_best_genome(best_genome, config, num_episodes=5)

    # Random baseline comparison
    test_random_baseline(
        num_episodes=5,
        max_steps=config['max_steps'],
        frame_skip=config['frame_skip']
    )

    # Ask if user wants to save (with custom name)
    print()
    save_choice = pick_menu(["Save best genome", "Don't save"])

    if save_choice == 0:
        print("\nEnter filename (without .pt extension): ", end='', flush=True)
        filename = input().strip()
        if filename:
            save_path = os.path.join(os.path.dirname(__file__), f'{filename}.pt')
            export_best_to_pytorch(best_genome, config, save_path)
        else:
            print("No filename entered, skipping save.")

    print("\nDone!")


if __name__ == '__main__':
    main()
