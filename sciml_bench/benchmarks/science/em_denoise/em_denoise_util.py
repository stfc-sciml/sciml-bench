#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# em_denoise_util.py

# SciML-Bench
# Copyright © 2021 Scientific Machine Learning Research Group
# Scientific Computing Department, Rutherford Appleton Laboratory
# Science and Technology Facilities Council, UK. 
# All rights reserved.

import math
import h5py
import torch
import numpy as np
import skimage.io
from pathlib import Path
from torch.utils.data import  DataLoader
from torch import nn, optim

from sciml_bench.core.utils import MultiLevelLogger, list_files


class EMDenoiseTrainingDataset(torch.utils.data.Dataset):
    """
    A generic zipped dataset loader for EMDenoiser
    """
    def __init__(self, noisy_file_path: Path, clean_file_path: Path):
        self.noisy_file_path = noisy_file_path
        self.clean_file_path = clean_file_path
        self.dataset_len = 0
        self.noisy_dataset = None
        self.clean_dataset = None

        with h5py.File(self.noisy_file_path, 'r') as hdf5_file:
            len_noisy = len(hdf5_file["images"])
        with h5py.File(self.clean_file_path, 'r') as hdf5_file:
            len_clean = len(hdf5_file["images"])

        self.dataset_len = min(len_clean, len_noisy)

    def __len__(self):
        return self.dataset_len
    
    def __getitem__(self, index):
        if self.noisy_dataset is None:
            self.noisy_dataset = h5py.File(self.noisy_file_path, 'r')["images"]
        if self.clean_dataset is None:
            self.clean_dataset = h5py.File(self.clean_file_path, 'r')["images"]
        return self.noisy_dataset[index], self.clean_dataset[index]


class EMDenoiseInferenceDataset(torch.utils.data.Dataset):
    """
    A inference dataset loader for EMDenoiser
    """
    def __init__(self, inference_file_path: Path, inference_gt_file_path: Path):
        self.inference_file_path = inference_file_path
        self.inference_gt_file_path = inference_gt_file_path
        self.dataset_len = 0
        self.inference_dataset = None
        self.inference_gt_dataset = None
        self.dataset_len = 0
        self.inference_images =  None
        self.inference_gt_images =  None
        self.inference_file_names =  list_files(path = self.inference_file_path, recursive=False)
        self.inference_gt_file_names =  list_files(path = self.inference_gt_file_path, recursive=False)
        self.dataset_len = min(len(self.inference_file_names), len(self.inference_gt_file_names))


    def __len__(self):
        return self.dataset_len
    
    def __getitem__(self, index):
        if self.inference_images == None:
            images =  np.zeros([ len(self.inference_file_names), 256, 256, 1], dtype=np.float32)
            for idx, url in enumerate(self.inference_file_names):
                images[idx, :, :, 0] =  skimage.io.imread(url)
            self.inference_images = torch.from_numpy( images.transpose((0, 3,1,2)) )
        if self.inference_gt_images == None:
            images =  np.zeros([ len(self.inference_gt_file_names), 256, 256, 1], dtype=np.float32)
            for idx, url in enumerate(self.inference_file_names):
                images[idx, :, :, 0] =  skimage.io.imread(url)
            self.inference_gt_images = torch.from_numpy( images.transpose((0, 3,1,2)) )
        
        return self.inference_images[index], self.inference_gt_images[index]


def train_model(log: MultiLevelLogger, model, train_loader: DataLoader, args, device):
    """
    Trains the EMDenoise AE Model. No validation. 
    """

    learning_rate = args['lr']
    epochs = args['epochs']

    model  = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    criterion = nn.MSELoss()
    
    train_history = []

    for epoch in range(epochs):
        running_loss = 0.0
        for batch_index, (noisy_batch, clean_batch) in enumerate(train_loader):
            # Transfer to GPU
            noisy_batch = torch.swapaxes(noisy_batch, 3, 1)
            clean_batch = torch.swapaxes(clean_batch, 3, 1)
            noisy_batch, clean_batch = noisy_batch.to(device), clean_batch.to(device)
            optimizer.zero_grad()
            ae_output = model(noisy_batch)
            train_loss = criterion(ae_output, clean_batch)
            train_loss.backward()
            optimizer.step()
            running_loss += train_loss.item()
        loss = running_loss / len(train_loader)
        train_history.append(loss)
        log.message(f'Epoch: {epoch}, loss: {loss}')

    return train_history

