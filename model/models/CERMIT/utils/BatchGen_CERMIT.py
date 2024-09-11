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


#Batch Generation for CERMIT net, (point cloud(1024 points) --- point cloud(2048 points))

shapenet_id_to_category = {
    '02691156': 'airplane',
    '02828884': 'bench',
    '02933112': 'cabinet',
    '02958343': 'car',
    '03001627': 'chair',
    '03211117': 'monitor',
    '03636649': 'lamp',
    '03691459': 'speaker',
    '04090263': 'rifle',
    '04256520': 'sofa',
    '04379243': 'table',
    '04401088': 'telephone',
    '04530566': 'vessel'
}

shapenet_category_to_id = {
    'airplane': '02691156',
    'bench': '02828884',
    'cabinet': '02933112',
    'car': '02958343',
    'chair'	: '03001627',
    'lamp'		: '03636649',
    'monitor'	: '03211117',
    'rifle'		: '04090263',
    'sofa'		: '04256520',
    'speaker'	: '03691459',
    'table'		: '04379243',
    'telephone'	: '04401088',
    'vessel'	: '04530566'
}

class BatchGen_CERMIT:

    def __init__(self, json_path, dots_directory):
        self.labels = []
        cats = shapenet_id_to_category.keys()

        with open(json_path, 'r') as f:
            train_models_dict = json.load(f)

        for cat in cats:
            self.labels.extend([model for model in train_models_dict[cat]])

        self.dots_directory = dots_directory

    def __len__(self):
        return len(self.labels)

    def transform_dots(self, dots):
        shape = dots.shape[0]

        if shape <= self.dots_size:

            for i in range(self.dots_size - shape):
                dots = pd.concat([dots, pd.DataFrame({'x': [0], 'y': [0], 'z': [0]})])
        else:
            indexes = np.random.choice(shape, shape - self.dots_size, replace=False)
            dots = dots.drop(indexes)

        return dots

    def normilize_dots(self, points):
        points -= np.min(points)
        points /= np.max(points)
        return points

    def __getitem__(self, index):
        label = self.labels[index]
        dots_path1 = os.path.join(self.dots_directory, os.path.join(label, 'pointcloud_1024.npy'))
        dots_path2 = os.path.join(self.dots_directory, os.path.join(label, 'pointcloud_2048.npy'))

        dots1 = np.load(dots_path1)
        dots2 = np.load(dots_path2)

        dots1 = self.normilize_dots(dots1)
        dots2 = self.normilize_dots(dots2)

        dots1 = torch.from_numpy(dots1).float()
        dots2 = torch.from_numpy(dots2).float()
        return dots1, dots2
