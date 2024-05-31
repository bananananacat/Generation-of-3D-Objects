import torch
import os
from random import randint
import numpy as np
import json

from .dots_loader import DotsLoader
from .image_loader import ImageLoader
from .shapenet_categories import shapenet_id_to_category


class BatchGen:
    
    def __init__(self, json_path, image_directory, dots_directory, max_size = 15, min_size = 5, dots_size = 2048, random = True):
        self.labels = []
        self.random = random
        self.max_size = max_size
        self.min_size = min_size
        self.image_directory = image_directory
        self.dots_directory = dots_directory
        self.dots_size = dots_size
        
        self.image_loader = ImageLoader()
        self.dots_loader = DotsLoader()
        
        cats = shapenet_id_to_category.keys()
        with open(json_path, 'r') as f:
            train_models_dict = json.load(f)

        for cat in cats:
            self.labels.extend([model for model in train_models_dict[cat]])

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        label = self.labels[index]
        dots_path = os.path.join(self.dots_directory, os.path.join(label, 'pointcloud_2048.npy'))
        image_path = os.path.join(self.image_directory, os.path.join(label, 'rendering'))
        images = None

        if self.random:
            num = len(os.listdir(image_path)) - 2
            size = randint(self.min_size, min(self.max_size, num)) 
            rnd = np.random.choice(num, size, replace=False)
            images = self.image_loader.open_image(os.path.join(image_path, str(rnd[0]).zfill(2) + '.png'))
            
            for i in range(1, len(rnd)):
                temp_path = os.path.join(image_path, str(rnd[i]).zfill(2) + '.png')
                images = torch.cat((images, self.image_loader.open_image(temp_path)))
            
            for i in range(self.max_size - size):
                images = torch.cat((images, torch.zeros((1, 3, 128, 128))))            
        else:
            images = self.image_loader.open_image(os.path.join(image_path, '0'.zfill(2) + '.png'))
            for i in range(1, self.max_size):
                temp_path = os.path.join(image_path, str(i).zfill(2) + '.png')
                images = torch.cat((images, self.image_loader.open_image(temp_path)))
            
        dots = self.dots_loader.open_dots(dots_path)

        return dots, images