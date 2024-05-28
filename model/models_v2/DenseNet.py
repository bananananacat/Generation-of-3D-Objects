class DenseNet(nn.Module):
    def __init__(self, input_size, upsampling_factor):
        super(DenseNet, self).__init__()
        self.upsampling_factor = upsampling_factor
        self.encoder = Encoder(input_size)
        self.decoder = Decoder(67)
      
    def forward(self, pcl):
        batch_size = pcl.shape[0]
        n = pcl.shape[1]
        # print(f'Input resolution: {self.in_pcl_size}; Output resolution: {self.out_pcl_size}')

        # Encoding the input point cloud
        z = self.encoder(pcl.permute(0, 2, 1))  # Adjust dimensions for Conv1d
        # print('z: ', z.shape)  # 'z: ', z.shape
        # print('pcl_in: ', pcl.shape)  # 'pcl_in: ', pcl.shape

        # Generate point features
        point_feat = pcl.unsqueeze(2).expand(-1, -1, self.upsampling_factor, -1)  # (bs,NUM_POINTS,3) --> (bs,NUM_POINTS,1,3) --> (bs,NUM_POINTS,2,3)
        point_feat = point_feat.reshape(batch_size, -1, 3)  # (bs,NUM_POINTS,2,3) --> (bs,NUM_UPSAMPLE_POINTS,3)
        # print('point_feat: ', point_feat.shape)

        # Generate global features
        global_feat = z.unsqueeze(1).expand(-1, self.upsampling_factor*n, -1)  # (bs,bneck) --> (bs,1,bneck) --> (bs,NUM_UPSAMPLE_POINTS,bneck)
        # print('global_feat: ', global_feat.shape)

        concat_feat = torch.cat([point_feat, global_feat], dim=2).permute(0, 2, 1)
        # print('concat_feat: ', concat_feat.shape)

        # Decoding the concatenated features
        dense = self.decoder(concat_feat)
        # print('dense1: ', dense.shape)
        dense = dense.view(batch_size, -1, 3)  # Reshape to (bs,NUM_UPSAMPLE_POINTS,3)
        # print('dense2: ', dense.shape)

        return dense


    def predict(self, x, upsample=2):
        for _ in range(upsample):
            x = self.forward(x)
        return x # [batch_size, N, 3] -> [batch_size, N * 2^upsample, 3]
