class CERMIT(nn.Module):
    def __init__(self, input_size, upsampling_factor, num_near_points, embed_size=64, device="cuda"):
        super(QualityNet, self).__init__()
        self.upsampling_factor = upsampling_factor
        self.device = device
        self.num_near_points = num_near_points

        self.encoder = Encoder(
            input_size=input_size,
            embed_size=embed_size,
            num_layers=6,
            heads=8,
            device=device,
            forward_expansion=2,
            dropout=0,
            num_near_points=num_near_points,
        )

        self.radius_encoder = Encoder(
            input_size=input_size + 3 * num_near_points,
            embed_size=embed_size,
            num_layers=6,
            heads=8,
            device=device,
            forward_expansion=2,
            dropout=0,
            num_near_points=num_near_points,
        )

        self.decoder = Decoder(
            input_size=3 + embed_size*4,
            embed_size=embed_size*2,
            num_layers=6,
            heads=8,
            forward_expansion=2,
            dropout=0,
            device=device,
        )

    def forward(self, pcl):
        batch_size = pcl.shape[0]
        n = pcl.shape[1]

        z = self.encoder(pcl)
        z = torch.mean(z, dim=1)
        point_feat = pcl.unsqueeze(2).expand(-1, -1, self.upsampling_factor, -1)
        point_feat = point_feat.reshape(batch_size, -1, 3)

        global_feat = z.unsqueeze(1).expand(-1, self.upsampling_factor * n, -1)

        concat_feat = torch.cat([point_feat, global_feat], dim=2)
        for i in range(3):
            near_points = get_nearest_points(pcl.cpu(), radius=0.03 + i / 100, num_near_points = self.num_near_points, device=self.device)
            data = concatenate_with_pcl(pcl, near_points, self.device)
            local_feat = self.radius_encoder(data)
            local_feat = local_feat.unsqueeze(2).repeat(1, 1, self.upsampling_factor, 1)
            local_feat = local_feat.view(batch_size, self.upsampling_factor * n, -1)
            concat_feat = torch.cat([concat_feat, local_feat], dim=2)

        dense = self.decoder(concat_feat)
        dense = dense.view(batch_size, -1, 3)
        return dense

    def predict(self, x, upsample=2):
        for _ in range(upsample):
            x = self.forward(x)
        return x
