import torch
from .image_loader import ImageLoader


class ValidLoader(ImageLoader):
    def __init__(self):
        super().__init__()
    
    def get_images(self, image_list):
        images = self.open_image(image_list[0])
        
        for i in range(1, len(image_list)):
            images = torch.cat((images, self.open_image(image_list[i])))

        return images