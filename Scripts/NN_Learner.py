import torch
import torch.nn as nn

class neural(nn.Module):
    def __init__(self, hidden_layers=8, neurons_per_layer=20):
        super(neural, self).__init__()

        # Input: (t, T, I) -> Output: Capacity C(t, T, I)
        layers = []
        layers.append(nn.Linear(3, neurons_per_layer))
        layers.append(nn.Tanh())

        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(neurons_per_layer, neurons_per_layer))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(neurons_per_layer, 1))  # Output layer predicts the Capacity C

        # Create sequential model
        self.network = nn.Sequential(*layers)

        # Xavier initialization for all Linear layers
        for m in self.network:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, t, T, I):
        inputs = torch.cat([t, T, I], dim=1)
        return self.network(inputs)

class ParameterLearner(nn.Module):
    def __init__(self):
        super().__init__()
        self.k  = nn.Parameter(torch.tensor(0.01, dtype=torch.float32))
        self.n  = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.Ea = nn.Parameter(torch.tensor(5000.0, dtype=torch.float32))  # J/mol typical scale

    def forward(self):
        return torch.stack([self.k, self.n, self.Ea])
    