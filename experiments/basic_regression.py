import torch
from torch import nn

# 1) Make synthetic data: y = 2x + 1 + noise
torch.manual_seed(0)
N = 100
X = torch.linspace(-1, 1, N).unsqueeze(1)          # shape [N, 1]
true_w, true_b = 2.0, 1.0
noise = 0.1 * torch.randn(N, 1)
y = true_w * X + true_b + noise                   # shape [N, 1]

# 2) Define a simple model: linear layer (1 input -> 1 output)
model = nn.Linear(in_features=1, out_features=1)

# 3) Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# 4) Training loop
for epoch in range(300):
    # Forward pass
    y_pred = model(X)
    loss = criterion(y_pred, y)

    # Backward + update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        w = model.weight.item()
        b = model.bias.item()
        print(f"Epoch {epoch+1:3d} | loss={loss.item():.4f} | w={w:.3f}, b={b:.3f}")

# 5) Final learned parameters
w = model.weight.item()
b = model.bias.item()
print(f"\nTrue w={true_w}, b={true_b}")
print(f"Learned w={w:.3f}, b={b:.3f}")
