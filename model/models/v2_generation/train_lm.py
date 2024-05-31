import torch 
import torch.nn as nn

from utils import ImageEncoder
from utils import EncoderDecoder
from utils import valid
from utils.loaders.batch_gen import BatchGen


if __name__ == "__main__":
    path1 = '../data/ShapeNetRendering/ShapeNetRendering'
    path2 = '../data/ShapeNet_pointclouds/ShapeNet_pointclouds'
    json_path = 'splits/train_models.json'
    json_path2 = 'splits/val_models.json'
    params = {'batch_size': 16, 'shuffle': True}
    num_epoch = 5
    counter = 0

    training_set = BatchGen(json_path, path1, path2)
    valid_set = BatchGen(json_path2, path1, path2)
    training_generator = torch.utils.data.DataLoader(training_set, **params)

    encoder_decoder = EncoderDecoder()
    encoder_decoder.load_state_dict(torch.load('trained_models/encoder_decoder.pt'))
    encoder_decoder.to('cuda')

    # turn off encoder grads
    for param in encoder_decoder.encoder.parameters():
        param.requires_grad = False

    image_encoder = ImageEncoder()
    image_encoder.to('cuda')

    mse_loss = nn.MSELoss()
    image_encoder_optim = torch.optim.Adam(image_encoder.parameters(), lr=1e-4)

    for i in range(num_epoch):
        for data, image in training_generator:
            counter += 1
            image_encoder_optim.zero_grad()        
            image = image.to('cuda')
            data = data.to('cuda')

            train_data = torch.transpose(data, 1, 2)
            out_base = encoder_decoder.encoder(train_data)

            # out_base shape will be [batch_size, 1024]
            out_image = image_encoder(image)
            loss = mse_loss(out_base, out_image)

            loss.backward()
            image_encoder_optim.step()
            
            if counter % 50 == 0:
                print(counter, loss)
                
            if counter % 100 == 0:
                print(valid(image_encoder, encoder_decoder.decoder, valid_set))
        
    torch.save(image_encoder.state_dict(), 'image_encoder.pt')