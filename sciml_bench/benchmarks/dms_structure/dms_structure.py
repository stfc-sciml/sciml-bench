#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# dms_structure.py

# SciML-Bench
# Copyright © 2021 Scientific Machine Learning Research Group
# Scientific Computing Department, Rutherford Appleton Laboratory
# Science and Technology Facilities Council, UK.
# All rights reserved.

import h5py
import yaml
import torch
import numpy as np
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder

from sciml_bench.core.runtime import RuntimeIn, RuntimeOut

from sciml_bench.benchmarks.dms_structure.dms_train import train_model
from sciml_bench.benchmarks.dms_structure.dms_model import DMSNet


# Helper Routine

def load_dms_datasets(dataset, device):
    hf = h5py.File(dataset, 'r')

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
        dataset_file = str(
            list(Path(params_in.dataset_dir).glob('**/data-binary.h5'))[0])
        datasets = load_dms_datasets(dataset_file, device)

    # create model
    with log.subproc('Creating the model'):
        model = DMSNet(device=device).to(device)

    # train the model
    with log.subproc('Training the model'):
        params_out.system.stamp_event('start training')
        history = train_model(log, model, datasets, args, params_in)

    # save model
    with log.subproc('Saving (entire) model to a file'):
        model_file = params_in.output_dir / f'dms_structure_model.h5'
        torch.save(model, model_file)

    # save history
    with log.subproc('Saving training history'):
        history_file = params_in.output_dir / 'training_history.yml'
        with open(history_file, 'w') as handle:
            yaml.dump(history, handle)

    log.ended(f'Running benchmark dms_structure on training mode')


#####################################################################
# Inference mode                                                    #
#####################################################################

def sciml_bench_inference(params_in: RuntimeIn, params_out: RuntimeOut):

    default_args = {
        'use_gpu': True
    }

    params_out.activate(rank=0, local_rank=0)

    log = params_out.log

    log.begin('Running benchmark dms_structure on inference mode')

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

    # save inference parameters
    args_file = params_in.output_dir / 'inference_arguments_used.yml'
    with log.subproc('Saving inference arguments to a file'):
        with open(args_file, 'w') as handle:
            yaml.dump(args, handle)

    # create datasets
    # with log.subproc('Loading inference dataset'):
        # inference_datasets = get_inference_data('')
    
    
    # Load the model and perform bulk inference
    with log.subproc('Loading the model for inference'):
        model = torch.load(params_in.model)
        model.to(device)
        model.eval()

    # More to come

    log.ended('Running benchmark dms_structure on inference mode')
