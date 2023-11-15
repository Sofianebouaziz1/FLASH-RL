import torch.nn as nn
import torch

class MLP(nn.Module):
    # RL module
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = torch.nn.ReLU()
        
    def forward(self, state):
        
        x = self.fc1(state)
        x = self.relu(x)

        # we dont want to activate it, we need the row numbers
        Q_values = self.fc2(x)
        
        return Q_values