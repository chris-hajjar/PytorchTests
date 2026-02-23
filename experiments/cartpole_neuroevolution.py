import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
import pickle
import os
import sys
from datetime import datetime

# Checkpoint path
CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), 'cartpole_neuroevo_checkpoint.pkl')

def pick_menu(options):
    """Arrow-key menu selector (reused from cartpole.py)."""
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
                sys.stdout.write(f"{prefix}{opt}\n")
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
    """Neural network for CartPole policy. Two architectures supported."""
    def __init__(self, state_size, action_size, hidden_size=128):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def select_action(self, state):
        """Deterministic policy: argmax over Q-values."""
        with torch.no_grad():
            q_values = self.forward(state)
            return q_values.argmax().item()


class SmallPolicyNetwork(nn.Module):
    """Smaller network for fast iteration (4→32→2)."""
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 32)
        self.fc2 = nn.Linear(32, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def select_action(self, state):
        """Deterministic policy: argmax over Q-values."""
        with torch.no_grad():
            q_values = self.forward(state)
            return q_values.argmax().item()


# ============================================================================
# GENOME ↔ NETWORK CONVERSION
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

def evaluate_genome(genome, env, model, device, k_episodes=1):
    """
    Evaluate a genome by running k episodes and averaging rewards.

    Args:
        genome: Genome object to evaluate
        env: Gymnasium environment
        model: PyTorch model to inject weights into
        device: torch device
        k_episodes: Number of episodes to average over

    Returns:
        Average fitness (total reward) across k episodes
    """
    inject_genome(model, genome.weights)
    model.eval()

    total_rewards = []

    for _ in range(k_episodes):
        state, _ = env.reset()
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        episode_reward = 0

        while True:
            action = model.select_action(state)
            observation, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward

            state = torch.FloatTensor(observation).unsqueeze(0).to(device)

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

    Args:
        population: List of Genome objects
        elite_count: Number of top genomes to preserve unchanged
        truncation_ratio: Fraction of population to keep as breeding pool

    Returns:
        elites, breeding_pool (both lists of Genome objects)
    """
    # Sort by fitness (descending)
    sorted_pop = sorted(population, key=lambda g: g.fitness, reverse=True)

    # Select elites (pass unchanged to next generation)
    elites = sorted_pop[:elite_count]

    # Select breeding pool (top X% for mutation)
    breeding_pool_size = max(elite_count, int(len(population) * truncation_ratio))
    breeding_pool = sorted_pop[:breeding_pool_size]

    return elites, breeding_pool


def mutate_genome(parent, mutation_std, generation):
    """
    Create offspring by adding Gaussian noise to parent weights.

    Args:
        parent: Parent Genome object
        mutation_std: Standard deviation of Gaussian noise
        generation: Current generation number

    Returns:
        New Genome object (child)
    """
    noise = np.random.normal(0, mutation_std, size=parent.weights.shape)
    child_weights = parent.weights + noise
    return Genome(child_weights, fitness=0.0, generation_born=generation)


def generate_offspring(breeding_pool, target_size, elite_count, mutation_std, generation):
    """
    Generate next generation from breeding pool.

    Args:
        breeding_pool: List of parent Genome objects
        target_size: Desired population size
        elite_count: Number of elites (already in next generation)
        mutation_std: Mutation strength
        generation: Current generation number

    Returns:
        List of offspring Genome objects
    """
    offspring = []
    num_offspring_needed = target_size - elite_count

    for _ in range(num_offspring_needed):
        parent = np.random.choice(breeding_pool)
        child = mutate_genome(parent, mutation_std, generation)
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


def export_best_to_pytorch(genome, config, save_path):
    """Export best genome as loadable PyTorch checkpoint."""
    # Create model matching config architecture
    if config['network_type'] == 'small':
        model = SmallPolicyNetwork(4, 2)
    else:
        model = PolicyNetwork(4, 2, hidden_size=128)

    # Inject genome weights
    inject_genome(model, genome.weights)

    # Save PyTorch checkpoint
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

def evolve(config, resume_from=None):
    """
    Main neuroevolution loop.

    Args:
        config: Dictionary with hyperparameters
        resume_from: Path to checkpoint file (if resuming)

    Returns:
        Best Genome object
    """
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make('CartPole-v1')  # No rendering during evolution

    # Create model architecture
    if config['network_type'] == 'small':
        model = SmallPolicyNetwork(4, 2).to(device)
    else:
        model = PolicyNetwork(4, 2, hidden_size=128).to(device)

    # Initialize or resume
    if resume_from and os.path.exists(resume_from):
        checkpoint = load_checkpoint(resume_from)
        population = checkpoint['population']
        start_gen = checkpoint['generation'] + 1
        history = checkpoint['history']
        print(f"Resumed from generation {checkpoint['generation']}")
    else:
        # Initialize random population
        print(f"Initializing random population of {config['population_size']}...")
        genome_size = sum(p.numel() for p in model.parameters())
        print(f"Genome size: {genome_size} parameters")

        population = []
        for i in range(config['population_size']):
            # Random initialization (similar to PyTorch default)
            random_weights = extract_genome(model)  # Get shape template
            random_weights = np.random.randn(len(random_weights)) * 0.1
            population.append(Genome(random_weights, fitness=0.0, generation_born=0))

        start_gen = 0
        history = {
            'best_fitness': [],
            'mean_fitness': [],
            'worst_fitness': []
        }

    # Evolution loop
    best_genome = None

    for generation in range(start_gen, config['max_generations']):
        print(f"\n{'='*60}")
        print(f"Generation {generation}")
        print(f"{'='*60}")

        # Evaluate population
        fitnesses = []
        for i, genome in enumerate(population):
            fitness = evaluate_genome(genome, env, model, device, config['fitness_episodes'])
            genome.fitness = fitness
            fitnesses.append(fitness)

            # Progress indicator
            if (i + 1) % 10 == 0 or (i + 1) == len(population):
                print(f"  Evaluated: {i+1}/{len(population)} genomes", end='\r')

        print()  # Newline after progress

        # Track statistics
        best_fitness = max(fitnesses)
        mean_fitness = np.mean(fitnesses)
        worst_fitness = min(fitnesses)

        history['best_fitness'].append(best_fitness)
        history['mean_fitness'].append(mean_fitness)
        history['worst_fitness'].append(worst_fitness)

        # Update best genome
        best_idx = np.argmax(fitnesses)
        if best_genome is None or fitnesses[best_idx] > best_genome.fitness:
            best_genome = population[best_idx]

        # Print stats
        print(f"  Best:  {best_fitness:6.1f}  |  Mean:  {mean_fitness:6.1f}  |  Worst:  {worst_fitness:6.1f}")

        # Check if solved (CartPole-v1 is "solved" at 500)
        if best_fitness >= 500:
            print(f"\n{'*'*60}")
            print(f"  SOLVED! Generation {generation} reached fitness {best_fitness:.1f}")
            print(f"{'*'*60}")
            # Continue for a few more generations to stabilize
            if generation >= start_gen + 10:
                break

        # Selection
        elites, breeding_pool = select_parents(
            population,
            config['elite_count'],
            config['truncation_ratio']
        )

        print(f"  Elites: {config['elite_count']}  |  Breeding pool: {len(breeding_pool)}")

        # Generate offspring
        offspring = generate_offspring(
            breeding_pool,
            config['population_size'],
            config['elite_count'],
            config['mutation_std'],
            generation + 1
        )

        # Create next generation (elites + offspring)
        population = elites + offspring

        # Save checkpoint every N generations
        if generation % config['checkpoint_interval'] == 0:
            save_checkpoint(population, generation, history, config, CHECKPOINT_PATH)
            print(f"  Checkpoint saved at generation {generation}")

    # Final checkpoint
    save_checkpoint(population, generation, history, config, CHECKPOINT_PATH)

    env.close()
    return best_genome, history


# ============================================================================
# TESTING
# ============================================================================

def test_best_genome(genome, config, num_episodes=10):
    """Test best genome."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create environment
    env = gym.make('CartPole-v1')

    # Create model
    if config['network_type'] == 'small':
        model = SmallPolicyNetwork(4, 2).to(device)
    else:
        model = PolicyNetwork(4, 2, hidden_size=128).to(device)

    # Load genome
    inject_genome(model, genome.weights)
    model.eval()

    print(f"\n{'='*60}")
    print(f"Testing best genome (Fitness: {genome.fitness:.1f}, Gen: {genome.generation_born})")
    print(f"{'='*60}")

    rewards = []
    for episode in range(num_episodes):
        state, _ = env.reset()
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        total_reward = 0

        while True:
            action = model.select_action(state)
            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            state = torch.FloatTensor(observation).unsqueeze(0).to(device)

            if terminated or truncated:
                break

        rewards.append(total_reward)
        print(f"  Episode {episode+1}: {total_reward:.0f}")

    print(f"\nAverage: {np.mean(rewards):.1f} ± {np.std(rewards):.1f}")

    env.close()
    return rewards


def test_random_baseline(num_episodes=10):
    """Test random agent for comparison."""
    env = gym.make('CartPole-v1')

    print(f"\n{'='*60}")
    print(f"Random baseline agent")
    print(f"{'='*60}")

    rewards = []
    for episode in range(num_episodes):
        env.reset()
        total_reward = 0

        while True:
            action = env.action_space.sample()
            _, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward

            if terminated or truncated:
                break

        rewards.append(total_reward)
        print(f"  Episode {episode+1}: {total_reward:.0f}")

    print(f"\nAverage: {np.mean(rewards):.1f} ± {np.std(rewards):.1f}")

    env.close()
    return rewards


# ============================================================================
# CONFIGURATION PROFILES
# ============================================================================

FAST_CONFIG = {
    'name': 'Fast iteration profile',
    'population_size': 50,
    'network_type': 'small',  # 4→32→2
    'fitness_episodes': 1,
    'mutation_std': 0.1,
    'elite_count': 2,
    'truncation_ratio': 0.2,  # Top 20% breed
    'max_generations': 200,
    'checkpoint_interval': 10,
}

STABLE_CONFIG = {
    'name': 'Stable learning profile',
    'population_size': 150,
    'network_type': 'large',  # 4→128→128→2
    'fitness_episodes': 5,
    'mutation_std': 0.05,
    'elite_count': 5,
    'truncation_ratio': 0.15,  # Top 15% breed
    'max_generations': 300,
    'checkpoint_interval': 10,
}


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    print("CartPole Neuroevolution\n")

    # Menu
    options = [
        f"New run - {FAST_CONFIG['name']} (~10 min)",
        f"New run - {STABLE_CONFIG['name']} (~60 min)",
    ]

    if os.path.exists(CHECKPOINT_PATH):
        options.append("Resume training (from checkpoint)")

    choice = pick_menu(options)

    # Determine config and resume status
    if choice == 0:
        config = FAST_CONFIG
        resume = False
    elif choice == 1:
        config = STABLE_CONFIG
        resume = False
    else:
        # Resume - load config from checkpoint
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
    print()

    # Run evolution
    best_genome, history = evolve(config, resume_from=CHECKPOINT_PATH if resume else None)

    # Print evolution summary
    print(f"\n{'='*60}")
    print(f"Evolution Summary")
    print(f"{'='*60}")
    print(f"Best fitness: {best_genome.fitness:.1f}")
    print(f"Best generation: {best_genome.generation_born}")
    print(f"Total generations: {len(history['best_fitness'])}")
    print(f"Final mean fitness: {history['mean_fitness'][-1]:.1f}")

    # Test best genome
    test_best_genome(best_genome, config, num_episodes=10)

    # Random baseline comparison
    test_random_baseline(num_episodes=10)

    # Ask if user wants to save
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
