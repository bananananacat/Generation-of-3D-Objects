import torch 
import kaolin

from utils import EncoderDecoder
from utils.loaders.batch_gen import BatchGen
from utils.losses import point_loss


if __name__ == "__main__":
    path1 = '../data/ShapeNetRendering/ShapeNetRendering'
    path2 = '../data/ShapeNet_pointclouds/ShapeNet_pointclouds'
    json_path = 'splits/train_models.json'
    params = {'batch_size': 64, 'shuffle': True}
    num_epoch = 20
    counter = 0
    points_loss_val = 0.001

    training_set = BatchGen(json_path, path1, path2)
    training_generator = torch.utils.data.DataLoader(training_set, **params)

    encoder_decoder = EncoderDecoder()
    encoder_decoder.to('cuda')

    optim = torch.optim.Adam(encoder_decoder.parameters(), lr=1e-4)

    for i in range(num_epoch):
        for data, _ in training_generator:
            counter += 1
            optim.zero_grad()        
            data = data.to('cuda')
            train_data = torch.transpose(data, 1, 2)
            
            out = encoder_decoder(train_data)
            out = out.reshape(-1, 2048, 3)
            
            loss = kaolin.metrics.pointcloud.chamfer_distance(out, data)
            loss = torch.mean(loss) + points_loss_val * point_loss(out)
            
            if counter % 100 == 0:
                print(counter, loss)
            
            loss.backward()
            optim.step()
            
    torch.save(encoder_decoder.state_dict(), 'trained_models/encoder_decoder.pt')
