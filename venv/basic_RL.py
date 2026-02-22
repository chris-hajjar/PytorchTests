import torch
import torch.nn.functional as F
from torch import nn

# 1) 3 slot machines with secret payout probabilities
true_probs = torch.tensor([0.1, 0.5, 0.9])  # arm 0 bad, 1 medium, 2 good

# 2) Model: neural net that outputs "action values" Q(s,a) for each arm
model = nn.Linear(1, 3)  # 1 input (dummy state) → 3 outputs (one per arm)

optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# 3) Training: 500 episodes
for episode in range(500):
    state = torch.tensor([[0.0]])  # dummy state (always same)
    
    # Forward: get Q values for each action
    q_values = model(state)  # shape [1, 3]
    
    # Epsilon-greedy: mostly pick best arm by model, sometimes random
    if torch.rand(1) < 0.1:  # 10% explore
        action = torch.randint(0, 3, (1,))
    else:  # 90% exploit model
        action = torch.argmax(q_values, dim=1)
    
    # Sample reward from true prob of chosen arm
    reward = torch.bernoulli(true_probs[action]).float()
    
    # Target for this action: reward + 0 (no future)
    target = reward
    
    # Loss: model prediction for chosen action vs actual reward
    pred = q_values.gather(1, action.unsqueeze(1)).squeeze()
    loss = F.mse_loss(pred, target.unsqueeze(0))
    
    # Update model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (episode + 1) % 100 == 0:
        best_arm = torch.argmax(q_values[0]).item()
        print(f"Ep {episode+1:3d} | Loss={loss.item():.3f} | Best arm={best_arm} | Reward={reward.item():.0f}")

# Final: print learned Q values vs true probs
print("\nLearned Q values:", q_values[0].detach().numpy())
print("True probs:      ", true_probs.numpy())
print("Best arm learned:", torch.argmax(q_values[0]).item())
