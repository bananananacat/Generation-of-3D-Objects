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



# This function finds 'num_near_points' nearest points for a given point cloud. This will be used for extracting local point features.
# If there are less then 'num_near_points' points inside a circle of radius=radius, returned ndarray will be filled with point(mean coordinates);
# it will contain mean_point * (num_near_points - num of points inside circle) times (for each points).

def get_nearest_points(pcl, radius=0.05, num_near_points=10, device="cuda"):
    points = None

    for step, pcl_ii in enumerate(pcl):
        pcl_i = pcl_ii.cpu()
        nbrs = NearestNeighbors(n_neighbors=num_near_points, radius=radius, algorithm='auto').fit(pcl_i.numpy())
        distances, indices = nbrs.radius_neighbors(pcl_i.numpy())

        n = pcl_i.shape[0]
        nearest_points_all = torch.zeros((n, num_near_points, 3))
        for i in range(n):
            nearest_points_within_radius = pcl_i[indices[i]]
            num_found_points = len(nearest_points_within_radius)

            if num_found_points >= num_near_points:
                sorted_indices = torch.tensor(distances[i]).argsort()[:num_near_points]
                nearest_points_all[i] = nearest_points_within_radius[sorted_indices]
                #perm = torch.randperm(num_near_points)
                #nearest_points_all[i] = nearest_points_all[i][perm]
            else:
                nearest_points_all[i, :num_found_points] = nearest_points_within_radius
                if num_found_points > 0:
                    mean_point = nearest_points_within_radius.mean(dim=0)
                    nearest_points_all[i, num_found_points:] = mean_point
                    perm = torch.randperm(num_near_points)
                    nearest_points_all[i] = nearest_points_all[i][perm]
        nearest_points_all = nearest_points_all.unsqueeze(0)
        if step == 0:
            points = nearest_points_all
        else:
            points = torch.cat((points, nearest_points_all), dim=0)

    return points


def concatenate_with_pcl(pcl, nearest_points_all, device):
    nearest_points_all_torch = nearest_points_all.to(device)
    nearest_points_all_flattened = nearest_points_all_torch.view(nearest_points_all_torch.shape[0],
                                                                 nearest_points_all_torch.shape[1], -1)
    result = torch.cat((pcl, nearest_points_all_flattened), dim=2)
    return result
