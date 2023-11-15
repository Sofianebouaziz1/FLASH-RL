import torch
import torch.nn as nn
import torch.nn.functional as F

class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
        super(GRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.GRU = nn.GRU(input_size = input_dim,hidden_size = hidden_dim, num_layers = n_layers, batch_first=True)
        self.linear1 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        h0 = self.init_hidden(x)
        out, hn = self.GRU(x, h0)
        out = self.linear1(self.relu(out[:, -1]))
        return out
    
    def init_hidden(self, x):
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim)
        return h0