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

#Batch Generation for PreCERMIT net, (image RGB (512, 512) --- point cloud(1024 points))

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

class BatchGen:

    def __init__(self, json_path, image_directory, dots_directory, dots_size=1024):
        self.num_imgs = 1
        self.labels = []
        self.image_directory = image_directory
        self.dots_directory = dots_directory
        self.dots_size = dots_size
        self.json_path = json_path
        cats = shapenet_id_to_category.keys()

        with open(self.json_path, 'r') as f:
            train_models_dict = json.load(f)

        for cat in cats:
            self.labels.extend([model for model in train_models_dict[cat]])

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
        points = torch.tensor(points).float()
        points -= torch.mean(points, dim=0)
        min_vals = torch.min(points, dim=0).values
        max_vals = torch.max(points, dim=0).values
        max_point = torch.max(-1 * min_vals, max_vals)
        points /= max_point
        return points

    def __getitem__(self, index):
        label = self.labels[index]
        png = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23"]
        for i in range(len(png)):
            png[i] += ".png"
        dots_path = os.path.join(self.dots_directory, os.path.join(label, 'pointcloud_1024.npy'))
        image_path = os.path.join(self.image_directory, label, 'rendering')

        transform_norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(torch.Tensor([0.0826, 0.0762, 0.0710]), torch.Tensor([0.1872, 0.1753, 0.1662]))
        ])
        n = random.randint(1, 24)
        new_image_path = os.path.join(image_path, png[n - 1])
        image = Image.open(new_image_path)
        image = image.convert('RGB')
        image = image.resize((512, 512))
        image = transform_norm(image)
        image = image.unsqueeze(0)

        for i in range(self.num_imgs - 1):
            n = random.randint(1, 24)
            new_image_path = os.path.join(image_path, png[n - 1])
            image_cat = Image.open(new_image_path)
            image_cat = image_cat.convert('RGB')
            image_cat = image_cat.resize((512, 512))
            image_cat = transform_norm(image_cat)
            image_cat = image_cat.unsqueeze(0)
            image = torch.cat((image, image_cat))

        dots = np.load(dots_path)
        dots = self.normilize_dots(dots)
        return image, dots
