#Decoder for DenseNet model

class Decoder(nn.Module):
    def __init__(self, input_size, layer_sizes=[512, 1024, 2048, 1024, 512, 3]):
        super().__init__()
        n_layers = len(layer_sizes)
        self.decoder = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for i in range(n_layers - 1):
            if i == 0:
                self.decoder.append(nn.Conv1d(input_size, layer_sizes[i], 2, stride=2))
            else:
                self.decoder.append(nn.Conv1d(layer_sizes[i - 1], layer_sizes[i], 2, stride=2))
                self.dropouts.append(nn.Dropout(0.3))
              
            self.decoder.append(nn.ConvTranspose1d(layer_sizes[i], layer_sizes[i], 2, stride=2))
            self.decoder.append(nn.ReLU())

        self.out = nn.Conv1d(layer_sizes[-2], layer_sizes[-1], 1)

    def forward(self, x):
        for layer in self.decoder:
            x = layer(x)
        x = self.out(x)
        return x

