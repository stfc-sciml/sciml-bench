
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# stemdl_classification.py

# SciML-Bench
# Copyright © 2021 Scientific Machine Learning Research Group
# Scientific Computing Department, Rutherford Appleton Laboratory
# Science and Technology Facilities Council, UK. 
# All rights reserved.

# Copyright © 2021 Oak Ridge National Laboratory
# P.O. Box 2008, Oak Ridge, TN 37831 USA. 
# All rights reserved.

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin

# imports from stemdl
import time
import sys
import os
import math
import glob
import argparse
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms, models
from tqdm import tqdm
from sklearn.metrics import f1_score
import torch.nn as nn
import numpy as np

from pathlib import Path
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor

# Integration into sciml-bench framework
from sciml_bench.core.runtime import RuntimeIn, RuntimeOut
from sciml_bench.core.utils import MultiLevelLogger
from sciml_bench.core.config import ProgramEnv
from sciml_bench.core.utils import SafeDict


# Custom dataset class
class NPZDataset(Dataset):
    def __init__(self, npz_root):
        self.files = glob.glob(npz_root + "/*.npz")

    def __getitem__(self, index):
        sample = np.load(self.files[index])
        x = torch.from_numpy(sample["data"])
        y = sample["label"][0]
        return (x, y)

    def __len__(self):
        return len(self.files)

# LitAutoEncoder
class LitAutoEncoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.input_size = 128
        self.num_classes = 231
        self.model_name = "resnet50"
        self.model = models.resnet50(pretrained=False)
        self.num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(self.num_ftrs, self.num_classes)
        self.feature_extract = False

    # forward step
    def forward(self, x):
        embedding = self.model(x)
        return embedding

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x_hat = self.model(x)
        y = F.one_hot(y, num_classes=231).float()
        loss = F.mse_loss(x_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x_hat = self.model(x)
        y = F.one_hot(y, num_classes=231).float()
        loss = F.mse_loss(x_hat, y)
        self.log('val_loss', loss)
        return loss

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        x_hat = self.model(x)
        y = F.one_hot(y, num_classes=231).float()
        loss = F.mse_loss(x_hat, y)
        self.log('test_loss', loss)
        return loss 
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y_hat = self.model(x)
        return y_hat

#####################################################################
# Training mode                                                     #
#####################################################################
# For training use this command:
# sciml-bench run stemdl_classification --mode training -b epochs 1 -b batchsize 32 -b nodes 1 -b gpus 1

def sciml_bench_training(params_in: RuntimeIn, params_out: RuntimeOut):

    # Entry point for the training routine to be called by SciML-Bench
    default_args = {
        'batchsize': 32,
        'epochs': 5,
        'nodes': 1,
        'gpus': 1
    }    

    # Log top level process
    log = params_out.log.console
    log.begin(f'Running benchmark stemdl_classification on training mode')

    # Parse input arguments against default ones 
    with log.subproc('Parsing input arguments'):
        args = params_in.bench_args.try_get_dict(default_args=default_args)
    
    # Set data paths
    with log.subproc('Set data paths'):
        basePath = params_in.dataset_dir
        # modelPath = params_in.output_dir / f'stemdlModel.h5'
        trainingPath =  os.path.expanduser(basePath / 'training')
        validationPath = os.path.expanduser(basePath / 'validation')
        testingPath = os.path.expanduser(basePath / 'testing')
        inferencePath = os.path.expanduser(basePath / 'inference')

    # Datasets: training (138717 files), validation (20000 files), 
    # testing (20000 files), prediction (8438 files), 197kbytes each
    with log.subproc('Create datasets'):
        train_dataset = NPZDataset(trainingPath)
        val_dataset = NPZDataset(validationPath)
        test_dataset = NPZDataset(testingPath)
        predict_dataset = NPZDataset(inferencePath)

    # Get command line arguments
    with log.subproc('Get command line arguments'):
        bs = int(args["batchsize"])
        epochs = int(args["epochs"])
        nodes = int(args["nodes"])
        gpus = int(args["gpus"])
    
    with log.subproc('Create datasets'):
        train_dataset = NPZDataset(trainingPath)
        val_dataset = NPZDataset(validationPath)
        test_dataset = NPZDataset(testingPath)
        predict_dataset = NPZDataset(inferencePath)

    # Create data loaders
    with log.subproc('Create data loaders'):
        train_loader = DataLoader(train_dataset, batch_size=bs, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=bs, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=bs, num_workers=4)
        #predict_loader = DataLoader(predict_dataset, batch_size=bs, num_workers=4)

    # Model
    with log.subproc('Create data model'):
        model = LitAutoEncoder()

    # Training
    with log.subproc('Start training'):
        trainer = pl.Trainer(gpus=gpus, num_nodes=nodes, precision=16, strategy="ddp", max_epochs=epochs)
        trainer.fit(model, train_loader, val_loader)
        log.message('End training')

    # Testing
    with log.subproc('Start testing'):
        trainer.test(model, test_loader)
        log.message('End testing')

    # Save model
    modelPathStr = '~/sciml_bench/outputs/stemdl_classification/stemdlModel.h5'
    modelPath = os.path.expanduser(modelPathStr)
    with log.subproc('Save model'):
        torch.save(model.state_dict(), modelPath)
        log.message('Model saved')

     # End top level
    log.ended(f'Running benchmark stemdl_classification on training mode')


#####################################################################
# Inference mode                                                    #
#####################################################################
# The "--model" key in inference commandline is not read.
'''
For inference run this command: 
sciml-bench run stemdl_classification --mode inference \
    --model ~/sciml_bench/outputs/stemdl_classification/stemdlModel.h5 \
    --dataset_dir ~/sciml_bench/datasets/stemdl_ds1 \
    -b epochs 1 -b batchsize 32 -b nodes 1 -b gpus 1 
'''
def sciml_bench_inference(params_in: RuntimeIn, params_out: RuntimeOut):
    # Entry point for the inference routine to be called by SciML-Bench
    log = params_out.log

    log.begin('Running benchmark stemdl_classification on inference mode')
    # Set data paths
    with log.subproc('Set data paths'):
        basePath = params_in.dataset_dir
        modelPathStr = '~/sciml_bench/outputs/stemdl_classification/stemdlModel.h5'
        modelPath = os.path.expanduser(modelPathStr)
        inferencePath = os.path.expanduser(basePath / 'inference')

    # Get command line arguments
    with log.subproc('Get command line arguments'):
        bs = int(params_in.bench_args["batchsize"])
        epochs = int(params_in.bench_args["epochs"])
        nodes = int(params_in.bench_args["nodes"])
        gpus = int(params_in.bench_args["gpus"])
    
    # Create datasets
    with log.subproc('Create datasets'):
        predict_dataset = NPZDataset(inferencePath)
    
    # Create data loaders
    with log.subproc('Create data loaders'):
        predict_loader = DataLoader(predict_dataset, batch_size=bs, num_workers=4)

    # Load model
    with log.subproc('Load model'):
        model = LitAutoEncoder()
        model.load_state_dict(torch.load(modelPath))

    # Start inference
    with log.subproc('Inference on the model'):
        trainer = pl.Trainer(gpus=gpus, num_nodes=nodes, precision=16, strategy="ddp")
        trainer.predict(model, dataloaders=predict_loader)

    # End top level
    log.ended('Running benchmark stemdl_classification on inference mode')
###
