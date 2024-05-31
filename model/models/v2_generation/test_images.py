import os
import torch 
import matplotlib.pyplot as plt

from utils import ImageEncoder
from utils import EncoderDecoder
from utils.loaders import ValidLoader


if __name__ == "__main__":
    files_path = '../data2'
    dots_encoder_decoder_path = 'trained_models/encoder_decoder.pt'
    image_encodert_path = 'trained_models/image_encoder.pt'
    
    encoder_decoder = EncoderDecoder()
    encoder_decoder.load_state_dict(torch.load(dots_encoder_decoder_path))
    encoder_decoder.to('cuda')
    encoder_decoder.eval()

    image_encoder = ImageEncoder()
    image_encoder.load_state_dict(torch.load('image_encodert_path'))
    image_encoder.to('cuda')

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')

    paths = [os.path.join(files_path, i) for i in os.listdir(files_path)]

    get_images = ValidLoader()

    images = get_images.get_images(paths)
    images = images.to('cuda')
    images = images.unsqueeze(0)

    out = image_encoder(images)
    out = encoder_decoder.decoder(out)

    out = out.reshape(-1, 2048, 3)
    out = out[0].cpu().detach().numpy().T

    ax.scatter(out[0], out[1], out[2], s = 1)
    plt.show()
