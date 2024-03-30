import os
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
import random


class BatchGen:
    
    def __init__(self, labels, image_directory, dots_directory):
        self.labels = labels
        self.image_directory = image_directory
        self.dots_directory = dots_directory
    
    
    def __len__(self):
        return len(self.labels)
    
    
    def __getitem__(self, index):
        id = self.labels[index]
        image_path = os.path.join(self.image_directory, f'scene{id}')
        dots_path = os.path.join(self.dots_directory, f'scene{id}.csv')
        out_images = torch.Tensor([])
        image_transformer = transforms.ToTensor()
        
        for file in os.listdir(image_path):
            image = Image.open(os.path.join(image_path, file))
            image = image.resize((224, 224))
            torch.cat((out_images, image_transformer(image)), 1)
        
        dots = pd.read_csv(dots_path).to_numpy()
        
        return out_images, dots
