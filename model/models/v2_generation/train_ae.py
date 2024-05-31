import torch 
import kaolin
import pickle

from utils import EncoderDecoder
from utils.loaders.batch_gen import BatchGen
from utils.losses import point_loss
from utils import valid_ae

from paths import *


if __name__ == "__main__":
    params = {'batch_size': 64, 'shuffle': True}
    num_epoch = 20
    counter = 0
    points_loss_val = 0.001
    valid_path = 'valid_data/valid_ae.pkl'

    training_set = BatchGen(json_path_train, data_images_path, data_points_path)
    valid_set = BatchGen(json_path_val, data_images_path, data_points_path)
    training_generator = torch.utils.data.DataLoader(training_set, **params)

    encoder_decoder = EncoderDecoder()
    encoder_decoder.to('cuda')

    optim = torch.optim.Adam(encoder_decoder.parameters(), lr=1e-4)

    valid_dict = {'step' : [], 'valid' : []}

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
            
            loss.backward()
            optim.step()
            
            if counter % 50 == 0:
                print(counter, loss)
            
            if counter % 100 == 0:
                valid_value = valid_ae(encoder_decoder, valid_set)
                print(counter, valid_value)
                valid_dict['step'].append(counter)
                valid_dict['valid'].append(valid_value)
    
    with open(valid_path, 'wb') as file:
        pickle.dump(valid_dict, file)        
    
    torch.save(encoder_decoder.state_dict(), dots_encoder_decoder_path)
