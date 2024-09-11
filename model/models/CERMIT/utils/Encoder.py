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

from utils.attention import *

class Encoder(nn.Module):
    def __init__(
        self,
        input_size,
        embed_size,
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
        num_near_points,
    ):

        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.embedding = nn.Linear(input_size, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size // heads,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                    device=device,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        N, seq_length, _ = x.shape
        out = self.embedding(x)

        for layer in self.layers:
            out = layer(out)

        return out
