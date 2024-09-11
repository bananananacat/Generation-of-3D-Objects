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

#our own loss function

def point_loss(points):
    loss = torch.empty(points.shape[0], dtype=torch.float)
    loss = loss.to('cuda')
    for i in range(points.shape[0]):
        dist = torch.cdist(points[i], points[i])
        dist = dist[(dist <= 0.01) & (dist >= 0.00001)]
        dist = 0.002 / dist
        loss[i] = dist.mean()
    loss = min(1.0, loss.mean() / 5.0)
    return loss
