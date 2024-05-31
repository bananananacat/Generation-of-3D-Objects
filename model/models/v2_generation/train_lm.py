import torch
import pickle
import torch.nn as nn

from loaders import BatchGen
from utils import ImageEncoder
from utils import EncoderDecoder
from utils import valid_lm

from paths import *


if __name__ == "__main__":
    params = {'batch_size': 16, 'shuffle': True}
    num_epoch = 5
    l_r = 1e-4
    counter = 0
    valid_path = 'valid_data/valid_lm.pkl'

    training_set = BatchGen(json_path_train, data_images_path, data_points_path)
    valid_set = BatchGen(json_path_val, data_images_path, data_points_path)
    training_generator = torch.utils.data.DataLoader(training_set, **params)

    encoder_decoder = EncoderDecoder()
    encoder_decoder.load_state_dict(torch.load(dots_encoder_decoder_path))
    encoder_decoder.to('cuda')

    # turn off encoder grads
    for param in encoder_decoder.encoder.parameters():
        param.requires_grad = False

    image_encoder = ImageEncoder()
    image_encoder.to('cuda')

    mse_loss = nn.MSELoss()
    image_encoder_optim = torch.optim.Adam(image_encoder.parameters(), lr=l_r)

    valid_dict = {'step' : [], 'valid' : []}

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
                valid_value = valid_lm(image_encoder, encoder_decoder.decoder, valid_set)
                print(counter, valid_value)
                valid_dict['step'].append(counter)
                valid_dict['valid'].append(valid_value)

    with open(valid_path, 'wb') as file:
        pickle.dump(valid_dict, file)
        
    torch.save(image_encoder.state_dict(), image_encoder_path)
