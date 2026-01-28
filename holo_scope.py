import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from holo_pytorch import HoloLinear, SineActivation # Assumes you saved the library

# ==============================================================================
# 1. THE DATA: A SPIRAL (Structure that requires 'Flow')
# ==============================================================================
def generate_spiral(n_points=1000):
    theta = np.sqrt(np.random.rand(n_points)) * 4 * np.pi # 720 degrees
    r_a = 2 * theta + np.pi
    data_a = np.array([np.cos(theta) * r_a, np.sin(theta) * r_a]).T
    x_a = data_a + np.random.randn(n_points, 2) * 1.5 # Add noise
    
    r_b = 2 * theta - np.pi
    data_b = np.array([np.cos(theta) * r_b, np.sin(theta) * r_b]).T
    x_b = data_b + np.random.randn(n_points, 2) * 1.5 # Add noise

    X = np.concatenate([x_a, x_b]) / 20.0 # Normalize
    Y = np.concatenate([np.zeros(n_points), np.ones(n_points)])
    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32).unsqueeze(1)

# ==============================================================================
# 2. THE CONTENDERS
# ==============================================================================
class StandardNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Standard Dense Layer: "I memorize lines"
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),           # The jagged cut
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

class HoloNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Holo Layer: "I resonate with curves"
        # Note: FAR FEWER PARAMETERS 
        # Standard: 2*64 + 64*64 + 64*1 = ~4300 params
        # Holo (16H): 16*2 + 16*64 + 16*64 + 16*1 = ~2100 params (Half the size)
        self.net = nn.Sequential(
            HoloLinear(2, 64, harmonics=16),
            SineActivation(),    # The phase lock
            HoloLinear(64, 64, harmonics=16),
            SineActivation(),
            HoloLinear(64, 1, harmonics=16),
            nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

# ==============================================================================
# 3. THE ARENA
# ==============================================================================
def train(model, X, Y, epochs=500):
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.BCELoss()
    for _ in range(epochs):
        loss = loss_fn(model(X), Y)
        opt.zero_grad(); loss.backward(); opt.step()
    return model

def visualize_decision_boundary(model, X, Y, title):
    # Create a grid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    
    # Predict
    with torch.no_grad():
        preds = model(grid).reshape(xx.shape)
    
    plt.contourf(xx, yy, preds, alpha=0.8, cmap='RdBu')
    plt.scatter(X[:, 0], X[:, 1], c=Y.flatten(), cmap='RdBu_r', edgecolors='k', s=20)
    plt.title(title)

# ==============================================================================
# 4. EXECUTION
# ==============================================================================
if __name__ == "__main__":
    X, Y = generate_spiral()
    
    print("Training Standard ReLU Network (The Memorizer)...")
    std_model = train(StandardNet(), X, Y)
    
    print("Training Holo Network (The Resonator)...")
    holo_model = train(HoloNet(), X, Y)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    visualize_decision_boundary(std_model, X, Y, "Standard ReLU (Jagged Cuts)")
    
    plt.subplot(1, 2, 2)
    visualize_decision_boundary(holo_model, X, Y, "Holo Phase (Resonant Flow)")
    
    plt.show()
    print("Look at the difference in the 'Blue/Red' boundary.")
    print("Standard = Polygons (Sharp). Holo = Waves (Smooth).")