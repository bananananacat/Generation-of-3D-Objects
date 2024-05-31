class Encoder(nn.Module):
    def __init__(self, input_size, n_filters=[32, 128, 512, 2048, 512, 128, 64]): #n_filters[-1], (n/2**n_filters >= 1)
        super().__init__()
        n_layers = len(n_filters)
        self.dropouts = nn.ModuleList([nn.Dropout(0.3) for i in range(n_layers)])
        self.encoder = nn.ModuleList(
            [nn.Conv1d(input_size, n_filters[i], 3, padding=1) if i == 0 else nn.Conv1d(n_filters[i - 1], n_filters[i], 3, padding=1) #k, k-1//2  padding = (kernel_size-1)/2
             for i in range(n_layers)])
        self.pools = nn.ModuleList([nn.MaxPool1d(2) for i in range(n_layers)])

    def forward(self, x):
        n = x.shape[2]
        for conv, pool, drop in zip(self.encoder, self.pools, self.dropouts):
            x = conv(x)
            x = drop(x)
            x = pool(x)
            x = F.relu(x)
        x = torch.mean(x, dim=2)
        return x
