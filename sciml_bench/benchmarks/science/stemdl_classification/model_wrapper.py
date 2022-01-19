import torch
import argparse
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms, models
import horovod.torch as hvd
import os
import math
from tqdm import tqdm
from npz_dataset import NPZDataset
import torch.nn as nn
import numpy as np

class TorchModel:
    def __init__(self, model_name="resnet50", input_size=128, num_classes=231, pretrained=False, feature_extract=False):
        self.model_name = model_name
        self.num_classes = num_classes
        
        self.model = None
        if self.model_name == "resnet18":
            self.model = models.resnet18(pretrained=pretrained)
        if self.model_name == "resnet34":
            self.model = models.resnet34(pretrained=pretrained)
        if self.model_name == "resnet50":
            self.model = models.resnet50(pretrained=pretrained)
        if self.model_name == "resnet100":
            self.model = models.resnet100(pretrained=pretrained)
 
        self.num_ftrs = self.model.fc.in_features   
        self.model.fc = nn.Linear(self.num_ftrs, self.num_classes)
        self.input_size = input_size
        self.params_to_update = self.model.parameters()
        self.feature_extract = feature_extract 
        
        if self.feature_extract:
            self.params_to_update = []
            for name,param in self.model.named_parameters():
                if param.requires_grad == True:
                    self.params_to_update.append(param)


