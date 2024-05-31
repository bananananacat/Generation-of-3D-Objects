import torch 
from PIL import Image
from torchvision import transforms


class ImageLoader:
    def __init__(self):
        self.transform_norm = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(torch.Tensor([0.0826, 0.0762, 0.0710]), torch.Tensor([0.1872, 0.1753, 0.1662]))
            ])

    def open_image(self, path):
        image = Image.open(path)
        image = image.convert('RGB')
        image = image.resize((128, 128))
        image = self.transform_norm(image).unsqueeze(0)
        
        return image
