#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# dms_structure.py

# SciML-Bench
# Copyright © 2021 Scientific Machine Learning Research Group
# Scientific Computing Department, Rutherford Appleton Laboratory
# Science and Technology Facilities Council, UK.
# All rights reserved.

import math
import time
import h5py
import yaml
import torch
import numpy as np
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder

from sciml_bench.core.runtime import RuntimeIn, RuntimeOut

from sciml_bench.benchmarks.science.dms_structure.dms_train import train_model
from sciml_bench.benchmarks.science.dms_structure.dms_model import DMSNet
from sciml_bench.benchmarks.science.dms_structure.dms_inference_dataset import DMSInferenceDataset


# Helper Routine

def load_dms_datasets(base_dataset_dir:Path, device, batch_size:int,  is_inference = False):

    if not is_inference:
        dataset_path = base_dataset_dir / 'training/data-binary.h5'
        hf = h5py.File(dataset_path, 'r')

        img = hf['train/images'][:]
        img = np.swapaxes(img, 1, 3)
        X_train = torch.from_numpy(np.atleast_3d(img)).to(device)
        lab = np.array(hf['train/labels']).reshape(-1, 1)
        onehot_encoder = OneHotEncoder(sparse=False)
        lab = onehot_encoder.fit_transform(lab).astype(int)
        Y_train = torch.from_numpy(lab).float().to(device)

        img = hf['test/images'][:]
        img = np.swapaxes(img, 1, 3)
        X_test = torch.from_numpy(np.atleast_3d(img)).to(device)
        lab = np.array(hf['test/labels']).reshape(-1, 1)
        lab = onehot_encoder.fit_transform(lab).astype(int)
        Y_test = torch.from_numpy(lab).float().to(device)

        datasets = (X_train, Y_train, X_test, Y_test)
        return datasets


    params = {
        'batch_size': batch_size,
        'shuffle': False,
        'num_workers': 2
    }

    if is_inference: 
        inference_path = base_dataset_dir 
        dms_inference_dataset = DMSInferenceDataset(inference_path)
        dms_inference_generator = torch.utils.data.DataLoader(dms_inference_dataset, **params)
        return dms_inference_generator



#####################################################################
# Training mode                                                     #
#####################################################################

def sciml_bench_training(params_in: RuntimeIn, params_out: RuntimeOut):
    default_args = {
        # hyperparameters
        'learning_rate': 0.0001,
        'epochs': 30,
        'batch_size': 32,
        'patience': 10,
        # workflow control
        'model_filename': 'model.pt',
        'validation_model_filename': 'valid_best_model.pt',
        'training_history': 'training_history.npy',
        'use_gpu': True
    }

    # initialise the system
    params_out.activate(rank=0, local_rank=0)
    log = params_out.log.console

    log.begin(f'Running benchmark dms_structure on training mode')

    # parse input arguments
    with log.subproc('Parsing input arguments'):
        args = params_in.bench_args.try_get_dict(default_args=default_args)

    # decide which device to use
    if args['use_gpu'] and torch.cuda.is_available():
        device = "cuda:0"
        log.message('Using GPU')
    else:
        device = "cpu"
        log.message('Using CPU')

    # save arguments
    args_file = params_in.output_dir / 'training_arguments_used.yml'
    with log.subproc('Saving arguments to a file'):
        with open(args_file, 'w') as handle:
            yaml.dump(args, handle)

    # load datasets
    with log.subproc('Loading datasets'):
        dataset_file = Path(params_in.dataset_dir) / 'data-binary.h5'
        datasets = load_dms_datasets(Path(params_in.dataset_dir), device, batch_size=args["batch_size"], is_inference=False)

    # create model
    with log.subproc('Creating the model'):
        model = DMSNet(device=device).to(device)

    # train the model
    with log.subproc('Training the model'):
        params_out.system.stamp_event('start training')
        start_time = time.time()
        history, metrics = train_model(log, model, datasets, args, params_in)
        end_time = time.time()
        time_taken = end_time - start_time

    # save model
    with log.subproc('Saving (entire) model to a file'):
        model_file = params_in.output_dir / f'dms_structure_model.h5'
        torch.save(model, model_file)

    # save history
    with log.subproc('Saving training history'):
        history_file = params_in.output_dir / 'training_history.yml'
        with open(history_file, 'w') as handle:
            yaml.dump(history, handle)

    # Save metrics
    metrics['time'] = time_taken
    metrics = {key: float(value) for key, value in metrics.items()}
    metrics_file = params_in.output_dir / 'metrics.yml'
    with log.subproc('Saving inference metrics to a file'):
        with open(metrics_file, 'w') as handle:
            yaml.dump(metrics, handle)  

    log.ended(f'Running benchmark dms_structure on training mode')


#####################################################################
# Inference mode                                                    #
#####################################################################

def sciml_bench_inference(params_in: RuntimeIn, params_out: RuntimeOut):

    """
    Entry point for the inference routine to be called by SciML-Bench
    """
    default_args = {
        'use_gpu': True,
        "batch_size" : 16
    }

    params_out.activate(rank=0, local_rank=0)

    log = params_out.log

    log.begin('Running benchmark dms_structure on inference mode')

    # Parse input arguments
    with log.subproc('Parsing input arguments'):
        args = params_in.bench_args.try_get_dict(default_args=default_args)

    # Decide which device to use
    if args['use_gpu'] and torch.cuda.is_available():
        device = "cuda:0"
        log.message('Using GPU for inference')
    else:
        device = "cpu"
        log.message('Using CPU for inference')

    # Save inference parameters
    args_file = params_in.output_dir / 'inference_arguments_used.yml'
    with log.subproc('Saving inference arguments to a file'):
        with open(args_file, 'w') as handle:
            yaml.dump(args, handle)  


    # Load the model and move it to the right device
    with log.subproc(f'Loading the model for inference'):
        model = torch.load(params_in.model)

    # Create datasets
    with log.subproc(f'Setting up a data loader for inference'):
        inference_dataset_loader = load_dms_datasets(params_in.dataset_dir, device, 
                                                    batch_size=args["batch_size"], is_inference=True)

    # Load the model and move it to the right device
    with log.subproc(f'Loading the model for inference into {device}'):
        model = torch.load(params_in.model)
        model.to(device)
    
    n_samples = len(inference_dataset_loader.dataset)

    # Perform bulk inference on the target device + collect metrics
    with log.subproc(f'Doing inference across {n_samples} items on device: {device}'):
        criterion = torch.nn.CrossEntropyLoss()
        start_time = time.time()
        total_loss  = 0
        # batch_correctness = torch.no_grad.Variable(batch_correctness).int()
        batch_correctness = 0
        for image_batch, label_batch in inference_dataset_loader:
            model.eval()
            with torch.no_grad():
                image_batch = image_batch.float().to(device)
                label_batch = label_batch.long().to(device)
                outputs = model.forward(image_batch)
                batch_correctness += torch.sum(outputs)
                total_loss += criterion(outputs, label_batch.flatten())
        rate = float(batch_correctness / n_samples) * 100
        loss = total_loss / n_samples
        end_time = time.time()
    time_taken = end_time - start_time

    throughput = math.floor(n_samples / time_taken)

    # Log outputs
    with log.subproc('Inference Performance'):
        log.message(f'Throughput  : {throughput:,} Images / sec')
        log.message(f'Overall Time: {time_taken:.4f} s')
        log.message(f'Correctness : {rate:.4f}%')

    # Save metrics
    metrics = dict(throughput=throughput, time=time_taken, accuracy=rate, loss=loss)
    metrics = {key: float(value) for key, value in metrics.items()}
    metrics_file = params_in.output_dir / 'metrics.yml'
    with log.subproc('Saving inference metrics to a file'):
        with open(metrics_file, 'w') as handle:
            yaml.dump(metrics, handle)  

    # End top level
    log.ended('Running benchmark dms_structure on inference mode')

   