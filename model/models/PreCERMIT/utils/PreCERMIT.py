#for this project: torch=1.12.1+cu116, torchaudio=0.12.1+cu116, torchvision=0.13.1+cu116
import kaolin

import random
import os
import re
import sys
import json
import time
import zipfile
import pickle
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange, Reduce
from sklearn.neighbors import NearestNeighbors
from transformers import get_cosine_with_hard_restarts_schedule_with_warmup

from utils.attention import *
from utils.ViTEncoder import *
from utils.Decoder import *
from utils.PreCERMIT import *

class PreCERMIT(nn.Module):
    def __init__(self, embed_size=64, device="cuda"):
        super(PreCERMIT, self).__init__()
        self.device = device

        self.encoder = ViTEncoder(
            image_size=512,
            patch_size=16,
            dim=embed_size,
            depth=12,
            heads=8
        )

        self.decoder = Decoder(
            input_size=embed_size,  # [batch_size, 16*16*3]
            embed_size=embed_size * 2,
            num_layers=12,
            heads=8,
            forward_expansion=2,
            dropout=0.2,
            device=device,
        )

    def forward(self, img):
        # print(img.shape)
        x = self.encoder(img)
        # print(x.shape)
        pcl = self.decoder(x)
        return pcl
