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

class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.transformer_block = TransformerBlock(
            embed_size,
            heads,
            dropout,
            forward_expansion,
            device=device,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.transformer_block(x)
        x = self.dropout(x)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        input_size,
        embed_size,
        num_layers,
        heads,
        forward_expansion,
        dropout,
        device,
    ):
        super(Decoder, self).__init__()
        self.device = device
        self.embedding = nn.Linear(input_size, embed_size)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size // heads,
                             heads,
                             forward_expansion,
                             dropout,
                             device)
                for _ in range(num_layers)
            ]
        )
        self.norms = nn.ModuleList(
            [
                nn.LayerNorm(embed_size)
                for _ in range(num_layers)
            ]
        )
        self.fc_out = nn.Linear(embed_size, 3)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        N, seq_length, _ = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        x = self.embedding(x)
        out = x
        for layer, norm in zip(self.layers, self.norms):
            out = layer(out) + x
            out = norm(out)

        out = self.fc_out(out)

        return out
