class DenseNet(nn.Module):
    def __init__(self, input_size, upsampling_factor): #upsampling_factor=2, num_radius_points=10
        super(DenseNet, self).__init__()
        self.upsampling_factor = upsampling_factor
        self.encoder = Encoder(input_size)
        self.radius_encoder = RadiusEncoder(input_size + 3 * 10)
        self.decoder = Decoder(259) #3 + 64 + 3 * 64

    def forward(self, pcl):
        batch_size = pcl.shape[0]
        n = pcl.shape[1]
        z = self.encoder(pcl.permute(0, 2, 1))
        
        point_feat = pcl.unsqueeze(2).expand(-1, -1, self.upsampling_factor, -1)  # (batch_size, n, 3) --> (batch_size, n, 1, 3) --> (batch_size, n, 2, 3)
        point_feat = point_feat.reshape(batch_size, -1, 3)  # (batch_size, n, upsampling_factor, 3) --> (batch_size, n * upsampling_factor,3)

        global_feat = z.unsqueeze(1).expand(-1, self.upsampling_factor*n, -1)  # (batch_size,bneck) --> (batch_size,1,bneck) --> 
        # --> (batch_size, n * upsampling_factor, bneck)

        concat_feat = torch.cat([point_feat, global_feat], dim=2).permute(0, 2, 1)
        
        # adding local point features (3 circles of radiuces 0.03, 0.04, 0.05)

        for i in range(3):
            near_points = get_nearest_points(pcl.cpu(), radius=0.03 + i/100, num_near_points=10)
            data = concatenate_with_pcl(pcl, near_points)
            data = data.transpose(1, 2)
            local_feat = self.radius_encoder(data)
            local_feat = local_feat.unsqueeze(2).repeat(1, 1, self.upsampling_factor, 1)  # (bs, n, 1, 64) --> (bs, n, 4, 64)
            local_feat = local_feat.view(batch_size, self.upsampling_factor*n, -1)  # (bs, n, 4, 64) --> (bs, n * upsampling_factor, 64)
            local_feat = torch.transpose(local_feat, 1, 2)

            concat_feat = torch.cat([concat_feat, local_feat], dim=1)

        dense = self.decoder(concat_feat)
        dense = dense.view(batch_size, -1, 3)  # Reshape to (bs, n * upsampling_factor, 3)

        return dense

    #function for generation pointcloud with  n * 2^upsample points
    def predict(self, x, upsample=2):
        for _ in range(upsample):
            x = self.forward(x)
        return x # [batch_size, n, 3] -> [batch_size, n * 2^upsample, 3]
