#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# em_denoise_util.py

# SciML-Bench
# Copyright © 2021 Scientific Machine Learning Research Group
# Scientific Computing Department, Rutherford Appleton Laboratory
# Science and Technology Facilities Council, UK. 
# All rights reserved.

import h5py
import torch
from pathlib import Path
from torch.utils.data import  DataLoader
from torch import nn, optim

from sciml_bench.core.utils import MultiLevelLogger


class EMDenoiseDataset(torch.utils.data.Dataset):
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
        for noisy_batch, clean_batch in train_loader:
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
