class PadiiNet(nn.Module):
    def __init__(self, embed_size=64, device="cuda"):
        super(PadiiNet, self).__init__()
        self.device = device

        self.encoder = ViTEncoder(
            image_size=512,
            patch_size=16,
            dim=embed_size,
            depth=12,
            heads=8
        )

        self.decoder = Decoder(
            input_size=embed_size,  # [batch_size, 16*16*3]
            embed_size=embed_size * 2,
            num_layers=12,
            heads=8,
            forward_expansion=2,
            dropout=0.2,
            device=device,
        )

    def forward(self, img):
        # print(img.shape)
        x = self.encoder(img)
        # print(x.shape)
        pcl = self.decoder(x)
        return pcl
