import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm 


class ImageEncoder(nn.Module):
    def __init__(self, hidden_size = 1024):
        super().__init__()
        #128 128
        self.conv_1 = nn.Conv2d(3, 32, 3, padding='same')
        self.conv_2 = nn.Conv2d(32, 32, 3, padding='same')
        self.conv_3 = nn.Conv2d(32, 64, 3, 2, padding='valid')
        
        self.conv_4 = nn.Conv2d(64, 64, 3, padding='same')
        self.conv_5 = nn.Conv2d(64, 64, 3, padding='same')
        self.conv_6 = nn.Conv2d(64, 128, 3, 2, padding='valid')
        
        self.conv_7 = nn.Conv2d(128, 128, 3, padding='same')
        self.conv_8 = nn.Conv2d(128, 128, 3, padding='same')
        self.conv_9 = nn.Conv2d(128, 256, 3, 2, padding='valid')
        
        self.conv_10 = nn.Conv2d(256, 256, 3, padding='same')
        self.conv_11 = nn.Conv2d(256, 256, 3, padding='same')
        self.conv_12 = nn.Conv2d(256, 512, 3, 2, padding='valid')
        
        self.conv_13 = nn.Conv2d(512, 512, 3, padding='same')
        self.conv_14 = nn.Conv2d(512, 512, 3, padding='same')
        self.conv_15 = nn.Conv2d(512, 512, 3, padding='same')
        
        self.conv_16 = nn.Conv2d(512, 512, 5, 2, padding='valid')
        
        self.out = nn.Linear(2048, hidden_size)

    def forward(self, x):
        batch_size = x.shape[0]
        image_size = x.shape[1]
        x = x.reshape(image_size * batch_size, 3, 128, 128)
        
        x = F.selu(self.conv_1(x))
        x = F.selu(self.conv_2(x))
        x = F.selu(self.conv_3(x))
        
        x = F.selu(self.conv_4(x))
        x = F.selu(self.conv_5(x))
        x = F.selu(self.conv_6(x))
        
        x = F.selu(self.conv_7(x))
        x = F.selu(self.conv_8(x))
        x = F.selu(self.conv_9(x))
        
        x = F.selu(self.conv_10(x))
        x = F.selu(self.conv_11(x))
        x = F.selu(self.conv_12(x))
        
        x = F.selu(self.conv_13(x))
        x = F.selu(self.conv_14(x))
        x = F.selu(self.conv_15(x))
        
        x = F.selu(self.conv_16(x))

        x = x.view(-1, 2048)
        
        x = self.out(x)        
        x = x.reshape(batch_size, image_size, -1)
        x = x.mean(dim=1)
                
        return x
