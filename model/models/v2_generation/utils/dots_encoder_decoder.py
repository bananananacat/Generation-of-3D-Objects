import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm 


class Encoder(nn.Module):
    def __init__(self, input_len = 2048, n_filters=[64, 128, 256, 512]):
        super().__init__()
        
        n_layers = len(n_filters)
        
        self.final_len = 126 * n_filters[-1]
        
        self.encoder = nn.ModuleList([nn.Conv1d(3, n_filters[i], 3) if i == 0 else nn.Conv1d(n_filters[i - 1], n_filters[i], 3)
                                    for i in range(n_layers)])

        self.pools = nn.ModuleList([nn.MaxPool1d(2) for i in range(n_layers)])

        self.out = nn.Linear(self.final_len, 1024)

    def forward(self, x):
        for layer in zip(self.encoder, self.pools):
            x = layer[0](x)
            x = layer[1](x)
            x = F.relu(x)

        x = x.view(-1, self.final_len)
        x = self.out(x)
        
        return x


class Decoder(nn.Module):
    def __init__(self, input_size, layer_sizes=[128, 256, 512, 1024, 2048, 2048 * 3]):
        super().__init__()
        
        n_layers = len(layer_sizes)
        
        self.decoder = nn.ModuleList([nn.Linear(input_size, layer_sizes[i]) if i == 0 else nn.Linear(layer_sizes[i - 1], layer_sizes[i])
                                    for i in range(n_layers - 1)])
        self.out = nn.Linear(layer_sizes[n_layers - 2], layer_sizes[n_layers - 1])
        
    def forward(self, x):
        for layer in self.decoder:
            x = layer(x)
            x = F.relu(x)
        
        x = self.out(x)
        
        return x


class EncoderDecoder(nn.Module):
    def __init__(self, hidden_size = 1024):
        super().__init__()
        
        self.encoder = Encoder()
        self.decoder = Decoder(hidden_size)
        
    def forward(self, x):
        
        x = self.encoder(x)
        x = self.decoder(x)
        
        return x
