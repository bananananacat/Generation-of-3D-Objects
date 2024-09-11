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

from utils.CustomDataLoader import *
from utils.attention import *
from utils.ViTEncoder import *
from utils.Decoder import *
from utils.PreCERMIT import *
from val import *
from utils.point_loss import *
from utils.BatchGen_PreCERMIT import *

def val(model, valid_generator):
    model.eval()
    metric1 = torch.tensor([0.0])
    metric1 = metric1.to('cuda')
    metric2 = torch.tensor([0.0])
    metric2 = metric2.to('cuda')
    num_examples = 0
    loss_mean = torch.tensor([0.0])
    loss_mean = loss_mean.to('cuda')
    with torch.no_grad():
        for img, pcl in valid_generator:
            model.zero_grad()
            img = img.to('cuda').unsqueeze(0)
            pcl = pcl.to('cuda').unsqueeze(0)
            pcl = pcl.view(-1, 2048, 3)
            out = model(img)
            loss_mean += torch.sum(kaolin.metrics.pointcloud.chamfer_distance(out, pcl))
            out = out.reshape(-1, 2048, 3)
            metric1 += torch.sum(kaolin.metrics.pointcloud.f_score(out, pcl))
            metric2 += torch.sum(kaolin.metrics.pointcloud.f_score(out, pcl, radius = 0.02))
            num_examples += out.shape[0]
            print(loss_mean)
    loss_mean /= num_examples
    metric1 /= num_examples
    metric2 /= num_examples
    return metric1, metric2, loss_mean
