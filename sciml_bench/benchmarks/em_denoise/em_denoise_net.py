
import numpy as np
import h5py
import yaml

import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
import mxnet.ndarray as nd
from mxnet import autograd as ag
from mxnet.gluon.utils import split_and_load



class EMDenoiseNet(gluon.Block):
    """ Network class """

    def __init__(self, decimation=1, **kwargs):
        super(EMDenoiseNet, self).__init__(**kwargs)

        def conv2d_3x3_same_padding_relu(channels):
            return nn.Conv2D(channels, kernel_size=(3, 3),
                             padding=(1, 1), activation='relu')

        def conv2d_trans_3x3_same_padding_relu(channels):
            return nn.Conv2DTranspose(channels, kernel_size=(3, 3),
                                      padding=(1, 1), activation='relu')

        base_channels = 8 if decimation <= 2 else 4
        with self.name_scope():
            # block 1
            self.conv11 = conv2d_3x3_same_padding_relu(base_channels)
            self.conv12 = conv2d_3x3_same_padding_relu(base_channels)
            self.bn11 = nn.BatchNorm()
            self.bn12 = nn.BatchNorm()
            self.mp1 = nn.MaxPool2D()
            # block 2
            self.conv21 = conv2d_3x3_same_padding_relu(base_channels * 2)
            self.conv22 = conv2d_3x3_same_padding_relu(base_channels * 2)
            self.bn21 = nn.BatchNorm()
            self.bn22 = nn.BatchNorm()
            self.mp2 = nn.MaxPool2D()
            # block 3
            self.conv31 = conv2d_3x3_same_padding_relu(base_channels * 4)
            self.conv32 = conv2d_3x3_same_padding_relu(base_channels * 4)
            self.bn31 = nn.BatchNorm()
            self.bn32 = nn.BatchNorm()
            self.mp3 = nn.MaxPool2D()
            # block 4
            self.conv41 = conv2d_3x3_same_padding_relu(base_channels * 8)
            self.conv42 = conv2d_3x3_same_padding_relu(base_channels * 8)
            self.bn41 = nn.BatchNorm()
            self.bn42 = nn.BatchNorm()
            # block 5
            self.conv51 = conv2d_trans_3x3_same_padding_relu(base_channels * 4)
            self.conv52 = conv2d_trans_3x3_same_padding_relu(base_channels * 4)
            self.bn51 = nn.BatchNorm()
            self.bn52 = nn.BatchNorm()
            # block 6
            self.conv61 = conv2d_trans_3x3_same_padding_relu(base_channels * 2)
            self.conv62 = conv2d_trans_3x3_same_padding_relu(base_channels * 2)
            self.bn61 = nn.BatchNorm()
            self.bn62 = nn.BatchNorm()
            # block 7
            self.conv71 = conv2d_trans_3x3_same_padding_relu(base_channels)
            self.conv72 = conv2d_trans_3x3_same_padding_relu(base_channels)
            self.bn71 = nn.BatchNorm()
            self.bn72 = nn.BatchNorm()
            # block 8, no activation
            self.conv8 = nn.Conv2D(1, kernel_size=(3, 3), padding=1)

    def forward(self, x):
        # skipped
        skipped = []
        # block 1
        x = self.bn12(self.conv12(self.bn11(self.conv11(x))))
        skipped.append(x)
        x = self.mp1(x)
        # block 2
        x = self.bn22(self.conv22(self.bn21(self.conv21(x))))
        skipped.append(x)
        x = self.mp2(x)
        # block 3
        x = self.bn32(self.conv32(self.bn31(self.conv31(x))))
        skipped.append(x)
        x = self.mp3(x)
        # block 4
        x = self.bn42(self.conv42(self.bn41(self.conv41(x))))
        # block 5
        x = nd.UpSampling(x, scale=2, sample_type='nearest')
        x = nd.concat(x, skipped.pop(-1))
        x = self.bn52(self.conv52(self.bn51(self.conv51(x))))
        # block 6
        x = nd.UpSampling(x, scale=2, sample_type='nearest')
        x = nd.concat(x, skipped.pop(-1))
        x = self.bn62(self.conv62(self.bn61(self.conv61(x))))
        # block 7
        x = nd.UpSampling(x, scale=2, sample_type='nearest')
        x = nd.concat(x, skipped.pop(-1))
        x = self.bn72(self.conv72(self.bn71(self.conv71(x))))
        # block 8
        x = self.conv8(x)
        return x

