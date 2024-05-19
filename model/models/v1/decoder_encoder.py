import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm 


class Encoder(nn.Module):
    def __init__(self, input_len, input_size, n_filters=[64, 128, 256, 512, 1024]):
        super().__init__()
        
        n_layers = len(n_filters)
        
        self.final_len = 64 * n_filters[-1]
        
        self.encoder = nn.ModuleList([weight_norm(nn.Conv1d(input_size, n_filters[i], 1), name='weight') if i == 0 else weight_norm(nn.Conv1d(n_filters[i - 1], n_filters[i], 1), name='weight')
                                    for i in range(n_layers)])
        
        self.dropouts = nn.ModuleList([nn.Dropout(0.2) for i in range(n_layers)])
        
        self.pools = nn.ModuleList([nn.MaxPool1d(2) for i in range(n_layers)])
        
        self.out = nn.Linear(self.final_len, 1024)

    def forward(self, x):
        for layer in zip(self.encoder, self.dropouts, self.pools):
            x = layer[0](x)
            x = layer[1](x)
            x = layer[2](x)
            x = F.selu(x)
        #print(x.shape)
        x = x.view(-1, self.final_len)
        x = self.out(x)
        
        return x


class Decoder(nn.Module):
    def __init__(self, input_size, layer_sizes=[64, 128, 256, 1024, 2048 * 3]):
        super().__init__()
        
        n_layers = len(layer_sizes)
        
        self.decoder = nn.ModuleList([weight_norm(nn.Linear(input_size, layer_sizes[i]), name='weight') if i == 0 else weight_norm(nn.Linear(layer_sizes[i - 1], layer_sizes[i]), name='weight')
                                    for i in range(n_layers - 1)])
        self.dropouts = nn.ModuleList([nn.Dropout(0.2) for i in range(n_layers - 1)])
        self.out = nn.Linear(layer_sizes[n_layers - 2], layer_sizes[n_layers - 1])
        
    def forward(self, x):
        for layer in zip(self.decoder, self.dropouts):
            x = layer[0](x)
            x = F.selu(x)
            x = layer[1](x)
        
        x = self.out(x)
        #x = torch.reshape(x, (-1, 2048, 3))
        
        return x
    

class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        #128 128
        self.conv_1 = nn.Conv2d(3, 32, 3, padding='same')
        self.conv_2 = nn.Conv2d(32, 32, 3, padding='same')
        self.conv_3 = nn.Conv2d(32, 64, 3, 2, padding='valid')
        
        self.conv_4 = nn.Conv2d(64, 64, 3, padding='same')
        self.conv_5 = nn.Conv2d(64, 64, 3, padding='same')
        self.conv_6 = nn.Conv2d(64, 128, 3, 2, padding='valid')
        
        self.conv_7 = nn.Conv2d(128, 128, 3, padding='same')
        self.conv_8 = nn.Conv2d(128, 128, 3, padding='same')
        self.conv_9 = nn.Conv2d(128, 256, 3, 2, padding='valid')
        
        self.conv_10 = nn.Conv2d(256, 256, 3, padding='same')
        self.conv_11 = nn.Conv2d(256, 256, 3, padding='same')
        self.conv_12 = nn.Conv2d(256, 512, 3, 2, padding='valid')
        
        self.conv_13 = nn.Conv2d(512, 512, 3, padding='same')
        self.conv_14 = nn.Conv2d(512, 512, 3, padding='same')
        self.conv_15 = nn.Conv2d(512, 512, 3, padding='same')
        
        self.conv_16 = nn.Conv2d(512, 512, 5, 2, padding='valid')
        
        self.out = nn.Linear(2048, 1024)

    def forward(self, x):
        #print(x.shape)
        x = F.selu(self.conv_1(x))
        x = F.selu(self.conv_2(x))
        x = F.selu(self.conv_3(x))
        
        #print(x.shape)
        x = F.selu(self.conv_4(x))
        x = F.selu(self.conv_5(x))
        x = F.selu(self.conv_6(x))
        
        #print(x.shape)
        x = F.selu(self.conv_7(x))
        x = F.selu(self.conv_8(x))
        x = F.selu(self.conv_9(x))
        
        #print(x.shape)
        x = F.selu(self.conv_10(x))
        x = F.selu(self.conv_11(x))
        x = F.selu(self.conv_12(x))
        
        #print(x.shape)
        x = F.selu(self.conv_13(x))
        x = F.selu(self.conv_14(x))
        x = F.selu(self.conv_15(x))
        
        x = F.selu(self.conv_16(x))
        #print(x.shape)

        x = x.view(-1, 2048)
        
        x = self.out(x)
        
        return x
