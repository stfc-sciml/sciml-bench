#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# dms_model.py

# SciML-Bench
# Copyright © 2021 Scientific Machine Learning Research Group
# Scientific Computing Department, Rutherford Appleton Laboratory
# Science and Technology Facilities Council, UK. 
# All rights reserved.

import torch.nn as nn
import torch.nn.functional as F

class DMSNet(nn.Module):
    """ Define a CNN """

    def __init__(self, device='cpu'):
        """ Initialize network with some hyperparameters """
        super(DMSNet, self).__init__()

        self.device = device
        self.conv1 = nn.Conv2d(3, 8, kernel_size=4)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=4)
        self.pool = nn.MaxPool2d(2)
        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(87584, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 2)
        self.drop = nn.Dropout(p=0.5)

        self.output_dim = 1

    def forward(self, x):
        x = self.bn1(self.pool(F.relu(self.conv1(x)))).to(self.device)
        x = self.bn2(self.pool(F.relu(self.conv2(x)))).to(self.device)
        x = x.view(x.size(0), -1).to(self.device)
        x = self.drop(F.relu(self.fc1(x))).to(self.device)
        x = self.drop(F.relu(self.fc2(x))).to(self.device)
        x = F.softmax(self.fc3(x), dim=1).to(self.device)
        return x

