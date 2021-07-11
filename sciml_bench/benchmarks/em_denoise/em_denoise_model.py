#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# em_denoise_net.py

# SciML-Bench
# Copyright © 2021 Scientific Machine Learning Research Group
# Scientific Computing Department, Rutherford Appleton Laboratory
# Science and Technology Facilities Council, UK. 
# All rights reserved.

import torch
import torch.nn as nn

class EMDenoiseNet(nn.Module):
    """ Define a CNN """
    def __init__(self, input_shape=(128,128,1)):

        super(EMDenoiseNet, self).__init__()
        self.input_shape = input_shape

        #encoder
        self.block1 = nn.ModuleList()
        self.block1.append(nn.Conv2d(self.input_shape[-1], 8, kernel_size=3, padding=1))
        self.block1.append(nn.ReLU())
        self.block1.append(nn.BatchNorm2d(8))
        self.block1.append(nn.Conv2d(8, 8, kernel_size=3, padding=1))
        self.block1.append(nn.ReLU())
        self.block1.append(nn.BatchNorm2d(8))
        self.block1.append(nn.MaxPool2d(2))

        self.block2 = nn.ModuleList()
        self.block2.append(nn.Conv2d(8, 16, kernel_size=3, padding=1))
        self.block2.append(nn.ReLU())
        self.block2.append(nn.BatchNorm2d(16))
        self.block2.append(nn.Conv2d(16, 16, kernel_size=3, padding=1))
        self.block2.append(nn.ReLU())
        self.block2.append(nn.BatchNorm2d(16))
        self.block2.append(nn.MaxPool2d(2))

        self.block3 = nn.ModuleList()
        self.block3.append(nn.Conv2d(16, 32, kernel_size=3, padding=1))
        self.block3.append(nn.ReLU())
        self.block3.append(nn.BatchNorm2d(32))
        self.block3.append(nn.Conv2d(32, 32, kernel_size=3, padding=1))
        self.block3.append(nn.ReLU())
        self.block3.append(nn.BatchNorm2d(32))
        self.block3.append(nn.MaxPool2d(2))

        self.block4 = nn.ModuleList()
        self.block4.append(nn.Conv2d(32, 64, kernel_size=3, padding=1))
        self.block4.append(nn.ReLU())
        self.block4.append(nn.BatchNorm2d(64))
        self.block4.append(nn.Conv2d(64, 64, kernel_size=3, padding=1))
        self.block4.append(nn.ReLU())
        self.block4.append(nn.BatchNorm2d(64))

        #decoder
        self.block5 = nn.ModuleList()
        self.block5.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        self.block5.append(nn.Conv2d(64 + 32, 32, kernel_size=3, padding=1))
        self.block5.append(nn.ReLU())
        self.block5.append(nn.BatchNorm2d(32))
        self.block5.append(nn.Conv2d(32, 32, kernel_size=3, padding=1))
        self.block5.append(nn.ReLU())
        self.block5.append(nn.BatchNorm2d(32))

        self.block6 = nn.ModuleList()
        self.block6.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        self.block6.append(nn.Conv2d(32 + 16, 16, kernel_size=3, padding=1))
        self.block6.append(nn.ReLU())
        self.block6.append(nn.BatchNorm2d(16))
        self.block6.append(nn.Conv2d(16, 16, kernel_size=3, padding=1))
        self.block6.append(nn.ReLU())
        self.block6.append(nn.BatchNorm2d(16))

        self.block7 = nn.ModuleList()
        self.block7.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        self.block7.append(nn.Conv2d(16 +8, 8, kernel_size=3, padding=1))
        self.block7.append(nn.ReLU())
        self.block7.append(nn.BatchNorm2d(8))
        self.block7.append(nn.Conv2d(8, 8, kernel_size=3, padding=1))
        self.block7.append(nn.ReLU())
        self.block7.append(nn.BatchNorm2d(8))

        self.last_layer = nn.Conv2d(8, 1, kernel_size=3, padding=1)



    def forward(self, x):
        skip_layers = []
        for i in range(len(self.block1)-1):
            x = self.block1[i](x)
        skip_layers.append(x)
        x = self.block1[-1](x)
        for i in range(len(self.block2)-1):
            x = self.block2[i](x)
        skip_layers.append(x)
        x = self.block2[-1](x)

        for i in range(len(self.block3)-1):
            x = self.block3[i](x)
        skip_layers.append(x)
        x = self.block3[-1](x)

        for i in range(len(self.block4)):
            x = self.block4[i](x)

        x = self.block5[0](x)
        x = torch.cat((x, skip_layers[-1]), dim=1)
        for i in range(len(self.block5)-1):
            x = self.block5[i+1](x)

        x = self.block6[0](x)
        x = torch.cat((x, skip_layers[-2]), dim=1)
        for i in range(len(self.block6)-1):
            x = self.block6[i+1](x)

        x = self.block7[0](x)
        x = torch.cat((x, skip_layers[-3]), dim=1)
        for i in range(len(self.block7)-1):
            x = self.block7[i+1](x)

        x = self.last_layer(x)
        return x

