#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# hydronet.py

# SciML-Bench
# Copyright © 2021 Scientific Machine Learning Research Group
# Scientific Computing Department, Rutherford Appleton Laboratory
# Science and Technology Facilities Council, UK. 
# All rights reserved.

'''
For running the hydronet benchmark we need to install dependencies with specific versions, otherwise
the application will not work. The dependencies can be installed by the following commands:

1)	Create conda environment
conda create --name hydronet2 python=3.8

2)	Activate conda environment
activate conda hydronet2

3)	Installing pytorch:
conda install pytorch==1.12.0 cudatoolkit=11.3 -c pytorch -c conda-forge

4)	conda install pyg -c pyg

5)	conda install -c conda-forge tensorboard ase fair-research-login h5py tqdm

6)	conda install -c conda-forge gdown

7) pip install pandas torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cu113.html

'''

import time
import yaml
import torch
import math
from pathlib import Path
import os, sys
import json, csv
import argparse, shutil
import logging
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from torch_geometric.nn import DataParallel
from tqdm import tqdm
import datetime

# Hydronet utils
import sciml_bench.benchmarks.science.hydronet.utils.data
import sciml_bench.benchmarks.science.hydronet.utils.models
import sciml_bench.benchmarks.science.hydronet.utils.train
import sciml_bench.benchmarks.science.hydronet.utils.infer
import sciml_bench.benchmarks.science.hydronet.utils.eval
import sciml_bench.benchmarks.science.hydronet.utils.split
import sciml_bench.benchmarks.science.hydronet.utils.hooks

# Dataloader dependencies
import os.path as op
from torch_geometric.data import DataListLoader, DataLoader
#from torch_geometric.loader import DataListLoader, DataLoader
from torch.utils.data import ConcatDataset
#from utils.datasets import PrepackedDataset
from sciml_bench.benchmarks.science.hydronet.utils.datasets import PrepackedDataset

# Data loading
import h5py
import random
from torch_geometric.data import DataListLoader, DataLoader, InMemoryDataset, Data, extract_zip, download_url

# Integration with SciML framework
from sciml_bench.core.runtime import RuntimeIn, RuntimeOut
from sciml_bench.core.utils import MultiLevelLogger
from sciml_bench.core.config import ProgramEnv
from sciml_bench.core.utils import SafeDict

# Libraries required for inference operation
import os.path as op
import pandas as pd
import json
import argparse
import matplotlib.pyplot as plt

# Inference methods
#
def force_magnitude_error(actual, pred):
    # ||f_hat|| - ||f||
    return torch.sub(torch.norm(pred, dim=1), torch.norm(actual, dim=1))

def force_angular_error(actual, pred):
    # cos^-1( f_hat/||f_hat|| • f/||f|| ) / pi
    # batched dot product obtained with torch.bmm(A.view(-1, 1, 3), B.view(-1, 3, 1))
    torch.pi = torch.acos(torch.zeros(1)).item() * 2 
    
    a = torch.norm(actual, dim=1)
    p = torch.norm(pred, dim=1)
    
    return torch.div(torch.acos(torch.bmm(torch.div(actual.T, a).T.view(-1, 1, 3), torch.div(pred.T, p).T.view(-1, 3,1 )).view(-1)), torch.pi)

# get_max_forces
def get_max_forces(data, forces):
    # data: data from DataLoader
    # forces: data.f for actual, f for pred
    start = 0
    f=[]
    for size in data.size.numpy()*3:
        f.append(np.abs(forces[start:start+size].numpy()).max())
        start += size
    return f

# infer
def infer(loader, net, forces=True, energies=True, force_type='max', device='cpu'):
    # force_type (str): ['max', 'all', 'error']
    f_actual = []
    e_actual = []
    e_pred = []
    f_pred = []
    size = []

    for data in loader:
        data = data.to(device)
        #size += data.size.tolist()
        size += data['size'].tolist()
        if energies:
            e_actual += data.y.tolist()
            e = net(data)
            e_pred += e.tolist()

        # get predicted values
        if forces:
            data.pos.requires_grad = True
            f = torch.autograd.grad(e, data.pos, grad_outputs=torch.ones_like(e), retain_graph=False)[0]
        
            if force_type == 'max':
                f_actual += get_max_forces(data, data.f)
                f_pred += get_max_forces(data, f)
            elif force_type == 'all':
                f_actual += data.f.squeeze().tolist()
                f_pred += f.squeeze().tolist()
            elif force_type == 'error':
                f_actual += force_magnitude_error(data.f, f)
                f_pred += force_angular_error(data.f, f)
            else:
                raise ValueError('Not implemented')
                
    # return as dataframe
    if energies == True and forces == False:
 
        # Relative difference between e_actual and e_pred
        num_e_actual = np.array(e_actual)
        num_e_pred = np.array(e_pred)
        num_error = abs((num_e_actual - num_e_pred)/num_e_actual)
        error = np.ndarray.tolist(num_error)
        
        return pd.DataFrame({'size': size, 'e_actual': e_actual, 'e_pred': e_pred, 'error': error})
    
    if energies == True and forces == True:
        if force_type=='error':
            return pd.DataFrame({'size': size, 'e_actual': e_actual, 'e_pred': e_pred}), pd.DataFrame({'f_mag_error': f_actual, 'f_ang_error': f_pred})
        else:
            return pd.DataFrame({'size': size, 'e_actual': e_actual, 'e_pred': e_pred}), pd.DataFrame({'f_actual': f_actual, 'f_pred': f_pred})

        
def infer_with_error(loader, net):
    fme_mse, fme_mae = [],[]
    fae_mse, fae_mae = [],[]
    e_actual = []
    e_pred = []
    f_pred = []
    size = []
    for data in loader:
        # extract ground truth values
        e_actual += data.y.tolist()
        size += data['size'].tolist()

        # get predicted values
        data.pos.requires_grad = True
        e = net(data)
        f = torch.autograd.grad(e, data.pos, grad_outputs=torch.ones_like(e), retain_graph=False)[0]

        # reshape f to ragged tensor
        # compute f errors for each sample
        start = 0
        for dsize in data['size'].numpy()*3:
            f_ragged = f[start:start+dsize]
            f_act = data.f[start:start+dsize]

            fme = force_magnitude_error(f_act, f_ragged)
            fae = force_angular_error(f_act, f_ragged)

            fme_mae += [torch.mean(torch.abs(fme)).tolist()]
            fme_mse += [torch.mean(torch.square(fme)).tolist()]
            fae_mae += [torch.mean(torch.abs(fae)).tolist()]
            fae_mse += [torch.mean(torch.square(fae)).tolist()]

            start += dsize

        # get properties
        e_pred += e.tolist()


    return pd.DataFrame({'size': size,
                         'e_actual': e_actual, 'e_pred': e_pred,
                         'fme_mae': fme_mae, 'fae_mae': fae_mae,
                         'fme_mse': fme_mse, 'fae_mse': fae_mse})


# init_dataloader_from_file
def init_dataloader_from_file(args, actionStr, split = '00', shuffle=True):
    """
    Returns train, val, and list of examine loaders
    """
    pin_memory = False if args.train_forces else True

    train_data = []
    val_data = []
    test_data = []
    
    # Read input file from args
    inputData = args.inputFile
    onlyFileName = inputData.split(".")[0]

    path1 = os.path.join(os.path.expanduser("~"), args.datadir)
    train_file = os.path.join(path1, onlyFileName+"_trainData.pt")
    val_file = os.path.join(path1, onlyFileName+"_valData.pt")
    test_file = os.path.join(path1, onlyFileName+"_testData.pt")

    if(actionStr =='train'):
        # load .pt files
        trainData = torch.load(train_file)
        valData = torch.load(val_file)

        train_data.append(trainData)
        val_data.append(valData)

        # no shuffle
        train_loader = DataLoader(ConcatDataset(train_data), batch_size=args.batch_size, shuffle=shuffle, pin_memory=pin_memory)
        val_loader = DataLoader(ConcatDataset(val_data), batch_size=args.batch_size, shuffle=shuffle, pin_memory=pin_memory)

        return train_loader, val_loader

    if(actionStr =='test'):
        testData = torch.load(test_file)
        test_data.append(testData)
        test_loader = DataLoader(ConcatDataset(test_data), batch_size=args.batch_size, shuffle=shuffle, pin_memory=pin_memory)
        return test_loader, testData

#####################################################################
# Training mode                                                     #
#####################################################################
#  sciml-bench run --mode training hydronet

def sciml_bench_training(params_in: RuntimeIn, params_out: RuntimeOut):
    """
    Entry point for the training routine to be called by SciML-Bench
    """
    # Framework level log
    log = params_out.log.console
    log.begin(f'Running benchmark hydronet on training mode')

    # Timestamp 
    start_time = datetime.datetime.now()

    # logging all to stdout
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    # Arguments file
    argsPath = os.path.join(os.getcwd(),"sciml_bench/benchmarks/science/hydronet")
    argsFile = os.path.join(argsPath,'train_args.json')

    # read in args from json file
    with open(argsFile) as f:
        args_dict = json.load(f)
        args = argparse.Namespace(**args_dict)
 
    ######## SET UP ########
    # create directory to store training results
    trainResultsDir = os.path.join(argsPath, 'trainingResults')
    args.savedir = trainResultsDir
    if not os.path.isdir(args.savedir):
        os.mkdir(args.savedir)
        os.mkdir(os.path.join(args.savedir,'tensorboard')) 
    else:
        logging.warning(f'{args.savedir} is already a directory, either delete or choose new SAVEDIR')
        sys.exit()
    
    # set up tensorboard logger
    writer = SummaryWriter(log_dir=os.path.join(args.savedir,'tensorboard'))
    
    # copy args file to training folder
    shutil.copy(argsFile, os.path.join(args.savedir, 'args.json'))

    # check for GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f'model will be trained on {device}')

    # increase batch size based on the number of GPUs
    if device == 'cuda' and args.parallel:
        n_gpus = torch.cuda.device_count()
        args.batch_size = int(n_gpus * args.batch_size)
        logging.info(f'... {n_gpus} found, multipying batch size accordingly (batch size now {args.batch_size})')
    
    ######## LOAD DATA ########
    # get initial train, val, examine splits for dataset(s)
    start_load = datetime.datetime.now()
    logging.info(f'Start loading data: {start_time}')
    train_loader, val_loader = init_dataloader_from_file(args, 'train')
    load_time = datetime.datetime.now() - start_load
    logging.info(f'Data load time: {load_time}')

    ######## LOAD MODEL ########
    net = sciml_bench.benchmarks.science.hydronet.utils.models.load_model(args, device=device)

    if device == 'cuda' and args.parallel: 
        net = DataParallel(net)
    logging.info(f'model loaded from {args.start_model}')

    #initialize optimizer and LR scheduler
    optimizer = torch.optim.Adam(net.parameters(), lr=args.start_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.8, min_lr=0.000001)

    # implement early stopping
    early_stopping = sciml_bench.benchmarks.science.hydronet.utils.hooks.EarlyStopping(patience=500, verbose=True, path = os.path.join(args.savedir, 'best_model.pt'),trace_func=logging.info)

    total_epochs = 0
    for _ in tqdm(range(args.n_epochs)):
        if args.train_forces:
            train_loss, e_loss, f_loss = sciml_bench.benchmarks.science.hydronet.utils.train.train_energy_forces(args, net, train_loader, optimizer, device)
            val_loss = sciml_bench.benchmarks.science.hydronet.utils.train.get_pred_loss(args, net, val_loader, optimizer, device)
        else:
            train_loss = sciml_bench.benchmarks.science.hydronet.utils.train.train_energy_only(args, net, train_loader, optimizer, device)
            val_loss = sciml_bench.benchmarks.science.hydronet.utils.train.get_pred_eloss(args, net, val_loader, optimizer, device)

        scheduler.step(val_loss)
    
        # log training info
        writer.add_scalars('epoch_loss', {'train':train_loss,'val':val_loss}, total_epochs)
        if args.train_forces:
            writer.add_scalars('train_loss', {'energy':e_loss,'forces':f_loss}, total_epochs)
        writer.add_scalar(f'learning_rate', optimizer.param_groups[0]["lr"], total_epochs)
        
        # check for stopping point
        early_stopping(val_loss, net)
        if early_stopping.early_stop:
            break

        total_epochs+=1

    # close tensorboard logger
    writer.close()

    # Program end timestamp
    run_time = datetime.datetime.now() - start_time

    logging.info(f'Hydronet: training complete: {run_time}')
    # End top level
    log.ended(f'Running benchmark hydronet on training mode')
    

#####################################################################
# Inference mode                                                    #
#####################################################################
# In the command line provide model file and dataset directory
# sciml-bench run --mode inference --model ~/swDev/sciml-bench-internal/sciml_bench/benchmarks/science/hydronet/trainingResults/best_model.pt --dataset_dir ~/sciml_bench/datasets/hydronet_ds1 hydronet
#
def sciml_bench_inference(params_in: RuntimeIn, params_out: RuntimeOut):
    """
    Entry point for the inference routine to be called by SciML-Bench
    """
    log = params_out.log.console
    log.begin(f'Running benchmark hydronet on inference mode')

    # logging all to stdout
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    # Arguments file
    argsPath = os.path.join(os.getcwd(),"sciml_bench/benchmarks/science/hydronet")
    argsFile = os.path.join(argsPath,'train_args.json')

    # read in args from json file
    with open(argsFile) as f:
        args_dict = json.load(f)
        args = argparse.Namespace(**args_dict)
 
    ######## SET UP ########
    # create directory to store training results
    trainResultsDir = os.path.join(argsPath, 'trainingResults')

    args.savedir = trainResultsDir
    args.load_state = True
    args.load_model = True
    args.start_model = op.join(args.savedir, 'best_model.pt')

    # check for GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load trained model
    net = sciml_bench.benchmarks.science.hydronet.utils.models.load_model(args, mode='eval', device=device, frozen=True)

    # load test data
    loader, data = init_dataloader_from_file(args, 'test')
    batch_size = args.batch_size if len(data) > args.batch_size else len(data)
    loader = DataLoader(data, batch_size=batch_size, shuffle=False)

    # get predictions on test set for each dataset
    df = pd.DataFrame()
    for dataset in args.datasets: 
        start_inference_time = datetime.datetime.now() 
        tmp = infer(loader, net, forces=args.train_forces, device=device)
        total_inference_time = datetime.datetime.now() - start_inference_time

        tmp['dataset']=dataset
        df = pd.concat([df, tmp], ignore_index=True, sort=False)
    
    # inference stats
    number_of_rows = len(df)
    logging.info(f'Inference time of {len(data)} queries: {total_inference_time}')
    logging.info(f'Time per inference in seconds: {total_inference_time/number_of_rows}')

    # save inference results
    df.to_csv(op.join(args.savedir, 'test_set_inference.csv'), index=False)

    logging.info('Hydronet: Inference complete!')
    
    # End top level
    log.ended('Running benchmark hydronet on inference mode')
