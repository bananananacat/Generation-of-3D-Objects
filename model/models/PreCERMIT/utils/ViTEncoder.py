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

class ViTEncoder(nn.Module):
    def __init__(self, image_size=512, patch_size=16, dim=768, depth=12, heads=8, device="cuda"):
        super(ViTEncoder, self).__init__()
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size ** 2

        self.patch_size = patch_size
        self.dim = dim
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.position_embeddings = nn.Parameter(torch.randn(1, num_patches, dim))

        self.transformer = nn.ModuleList([TransformerBlock(dim_head=dim // heads, heads=heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, img):
        p = self.patch_size
        bs, cnt, c, h, v = img.shape

        img = img.reshape(-1, 3, 512, 512)
        print(img.shape, "img")
        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        x = self.patch_to_embedding(x)
        x += self.position_embeddings
        for block in self.transformer:
            x = block(x)

        x = self.norm(x)
        print(x.shape, "do mean")
        x = x.reshape(bs, cnt, 1024, -1)
        print(x.shape, "do mean, reshape")
        x = x.mean(dim=1)
        print(x.shape, "posle mean")
        return x
