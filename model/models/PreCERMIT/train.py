#for this project: torch=1.12.1+cu116, torchaudio=0.12.1+cu116, torchvision=0.13.1+cu116
#kaolin lib works with 'cuda'
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


if __name__ == "__main__":

    save_dir = 'path to save model directory'
    save_nums_dir = 'path to save dicts with nums directory'

    pcl_path = 'path to point cloud dataset(we used ShapeNet, citing is in paper)'
    json_path = 'path to splits dataset(we used ShapeNet, citing is in paper)'

    img_path = "path to images dataset(we used ShapeNet, citing is in paper)"

    val_pcl_path = 'path to validation dataset / path to point cloud dataset(we used ShapeNet, citing is in paper)'
    val_json_path = 'path to validation dataset / path to splits dataset(we used ShapeNet, citing is in paper)'

    params_train = {'batch_size': 4, 'shuffle': True}
    params_val = {'batch_size': 4, 'shuffle': False}

    training_generator = CustomDataLoader(json_path, img_directory=img_path, dots_directory=pcl_path, **params_train)
    valid_generator = CustomDataLoader(json_path, img_directory=img_path, dots_directory=pcl_path, **params_val)

    precermit = PreCERMIT()
    precermit = precermit.to('cuda')

    optim = torch.optim.Adam(precermit.parameters(), lr=1e-4)
    #scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optim, num_warmup_steps= 150, num_training_steps=len(training_generator), num_cycles=2)

    points_loss_val = 0.01
    counter_ep = 1
    counter = 0

    loss_dict = {}
    loss_val_dict = {}
    val_dict1 = {}
    val_dict2 = {}

    for i in range(20):
        print(f"{counter_ep} epoch \n")
        for img, pcl in training_generator:
            counter += 1
            optim.zero_grad()
            img = img.to('cuda')
            pcl = pcl.to('cuda')
            print(img.shape, pcl.shape)
            out = precermit(img)
            out = out.reshape(-1, 1024, 3)
            pcl = pcl.view(-1, 1024, 3)
            loss = kaolin.metrics.pointcloud.chamfer_distance(out, pcl)
            ch = torch.mean(loss)
            plv = point_loss(out)
            loss = ch + points_loss_val * plv
            if counter % 100 == 0:
                print(counter, loss.item())
                if counter % 300 == 0:
                    current_lr = optim.param_groups[0]['lr']
                    print(f"Step: {counter}, LR: {current_lr}         ")
                loss_dict[counter] = loss.item()

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(precermit.parameters(), clip_value=1.0)
            optim.step()
            #scheduler.step()

        score1, score2, loss_mean = val(precermit, valid_generator)
        print(f"epoch = {counter_ep}, score1 = {score1.item()}, score2 = {score2.item()}, loss_val_mean = {loss_mean.item()}")
        val_dict1[counter_ep] = score1.item()
        val_dict2[counter_ep] = score2.item()
        loss_val_dict[counter_ep] = loss_mean.item()
        torch.save(precermit.state_dict(), os.path.join(save_dir, f'precermit_{counter_ep}_epoch.pt'))

        precermit.train()
        counter_ep += 1

    print("end of learning")
    torch.save(precermit.state_dict(), 'precermit.pt')

    with open(os.path.join(save_nums_dir, 'loss'), 'wb') as f:
        pickle.dump(loss_dict, f)
    with open(os.path.join(save_nums_dir, 'val1'), 'wb') as f:
        pickle.dump(val_dict1, f)
    with open(os.path.join(save_nums_dir, 'val2'), 'wb') as f:
        pickle.dump(val_dict2, f)
    with open(os.path.join(save_nums_dir, 'loss_val'), 'wb') as f:
        pickle.dump(loss_val_dict, f)
