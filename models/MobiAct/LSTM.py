import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
        # batch_first = true, nos données sont de taille = (batch_size, seq, input_size)
        self.linear1 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.i = 0
        

    def forward(self, x):
        h0, c0 = self.init_hidden(x) #initialiser la mémoire pour chaque entrée
        out, (hn, cn) = self.lstm(x, (h0, c0)) #appliquer LSTM
        # out est de taille (batch, seq_length, hidden_size)
        # on veut seulement la classification du dernier timestep dans la sequence -> (batch, hidden_size)
        out = self.linear1(self.relu(out[:, -1])) #on applique relu pour le out avant la sortie
        return out
    
    def init_hidden(self, x):
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_dim)
        return [t for t in (h0, c0)]