import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
from collections import deque
import random
import os

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), 'cartpole_checkpoint.pt')

def pick_menu(options):
    """Arrow-key menu selector."""
    import sys, tty, termios
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

print("CartPole DQN Training")
options = ["New run (start fresh)"]
if os.path.exists(CHECKPOINT_PATH):
    options.append("Resume training (from checkpoint)")
choice = pick_menu(options)
resume = choice == 1

# Hyperparams
GAMMA = 0.99
EPS_START, EPS_END, EPS_DECAY = 1.0, 0.02, 500
BATCH_SIZE = 128
LEARNING_RATE = 1e-4
MEMORY_SIZE = 10000
TARGET_UPDATE = 10

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Replay buffer
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

# Create env with live rendering
env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]  # 4
action_size = env.action_space.n  # 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy_net = DQN(state_size, action_size).to(device)
target_net = DQN(state_size, action_size).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
memory = ReplayMemory(MEMORY_SIZE)

steps_done = 0
start_episode = 0

if resume and os.path.exists(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    policy_net.load_state_dict(checkpoint['policy_net'])
    target_net.load_state_dict(checkpoint['target_net'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    steps_done = checkpoint['steps_done']
    start_episode = checkpoint['episode'] + 1
    print(f"Resumed from episode {start_episode} (steps: {steps_done})")

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        np.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*transitions)

    state_batch = torch.cat(states).to(device)
    action_batch = torch.cat(actions).to(device)
    reward_batch = torch.tensor(rewards, device=device, dtype=torch.float32)
    next_state_batch = torch.cat(next_states).to(device)
    done_batch = torch.tensor(dones, device=device, dtype=torch.bool)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[~done_batch] = target_net(next_state_batch[~done_batch]).max(1)[0]

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

# TRAINING
scores = []
for i_episode in range(start_episode, start_episode + 1000):
    state, _ = env.reset()
    state = torch.FloatTensor(state).unsqueeze(0).to(device)
    total_reward = 0
    
    while True:
        action = select_action(state)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        total_reward += reward
        
        next_state = torch.FloatTensor(observation).unsqueeze(0).to(device)
        done = terminated or truncated
        
        memory.push(state, action, reward, next_state, done)
        
        state = next_state
        
        optimize_model()
        
        if done:
            break
    
    scores.append(total_reward)
    eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps_done / EPS_DECAY)
    print(f"Episode {i_episode}, Score: {total_reward:.0f}, Epsilon: {eps_threshold:.3f}")

    
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    torch.save({
        'policy_net': policy_net.state_dict(),
        'target_net': target_net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'steps_done': steps_done,
        'episode': i_episode,
    }, CHECKPOINT_PATH)

print("Training done!")

# TEST TRAINED AGENT (watch it balance!)
print("\n=== Testing trained agent ===")
state, _ = env.reset()
state = torch.FloatTensor(state).unsqueeze(0).to(device)
total_reward = 0
while True:
    with torch.no_grad():
        action = policy_net(state).max(1).indices.view(1, 1)
    observation, reward, terminated, truncated, _ = env.step(action.item())
    total_reward += reward
    state = torch.FloatTensor(observation).unsqueeze(0).to(device)
    if terminated or truncated:
        break

print(f"Trained score: {total_reward}")

# RANDOM AGENT COMPARISON
print("\n=== Random agent (baseline) ===")
state, _ = env.reset()
total_reward = 0
while True:
    action = env.action_space.sample()
    observation, reward, terminated, truncated, _ = env.step(action)
    total_reward += reward
    if terminated or truncated:
        break

print(f"Random score: {total_reward}")

env.close()
