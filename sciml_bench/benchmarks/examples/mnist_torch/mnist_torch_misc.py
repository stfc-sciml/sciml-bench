#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# impl_mnist_torch.py

# SciML-Bench
# Copyright © 2021 Scientific Machine Learning Research Group
# Scientific Computing Department, Rutherford Appleton Laboratory
# Science and Technology Facilities Council, UK.
# All rights reserved.

"""
Various routes to be used by the mnist_torch benchmark
"""

import torch
import torch.utils.data as Data
import torch.nn as nn
import torch.nn.functional as F
import h5py


class MNISTDataset(Data.Dataset):
    """mnist dataset"""

    def __init__(self, file_path):
        super(MNISTDataset, self).__init__()
        # read whole
        with h5py.File(file_path, "r") as h5:
            self.images = torch.tensor(h5["image"][:], dtype=torch.float32)
            self.labels = torch.tensor(h5["label"][:], dtype=torch.long)
        # expand channel dim and normalize
        self.images = torch.unsqueeze(self.images, 1) / 255

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.images.size(0)


class MNISTNet(nn.Module):
    """Define a CNN"""

    def __init__(self, n_filters, n_units_hidden):
        """Initialize network with some hyperparameters"""
        super(MNISTNet, self).__init__()
        self.conv = nn.Conv2d(1, n_filters, kernel_size=3)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(n_filters * 13 * 13, n_units_hidden)
        self.fc2 = nn.Linear(n_units_hidden, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv(x)))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        x = F.softmax(self.fc2(x), dim=1)
        return x
