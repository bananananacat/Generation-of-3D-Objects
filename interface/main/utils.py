import os
import sys
import torch
sys.path.append("../../model/models/v2_generation")
from utils import EncoderDecoder
from utils import ImageEncoder
from loaders import ImageLoader


path_encoder_decoder = "../../model/models/v2_generation/trained_models/encoder_decoder.pt"
path_image_encoder = "../../model/models/v2_generation/trained_models/image_encoder.pt"

def convert_photos_to_3d_model(upload_path):
    encoder_decoder = EncoderDecoder()
    encoder_decoder.load_state_dict(torch.load(path_encoder_decoder))
    image_decoder = ImageEncoder()
    image_decoder.load_state_dict(torch.load(path_image_encoder))
    
    model_path = os.path.join(upload_path, 'result.obj')
    success = True
    return success, model_path
