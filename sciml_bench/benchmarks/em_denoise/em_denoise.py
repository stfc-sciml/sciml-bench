#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# em_denoise.py

# SciML-Bench
# Copyright © 2021 Scientific Machine Learning Research Group
# Scientific Computing Department, Rutherford Appleton Laboratory
# Science and Technology Facilities Council, UK. 
# All rights reserved.


import yaml
import torch
from pathlib import Path

from sciml_bench.core.runtime import RuntimeIn, RuntimeOut

from sciml_bench.benchmarks.em_denoise.em_denoise_util import EMDenoiseDataset,train_model
from sciml_bench.benchmarks.em_denoise.em_denoise_model import EMDenoiseNet


def get_train_data_generator(base_dataset_dir: Path, batch_size: int):

    params = {
        'batch_size': batch_size,
        'shuffle': True,
        'num_workers': 2
    }

    noisy_path = str(base_dataset_dir / 'graphene_img_noise.h5')
    clean_path = str(base_dataset_dir / 'graphene_img_clean.h5')
    em_denoise_dataset = EMDenoiseDataset(noisy_path, clean_path)
    em_denoise_generator = torch.utils.data.DataLoader(em_denoise_dataset, **params)

    return em_denoise_generator



#####################################################################
# Training mode                                                     #
#####################################################################

def sciml_bench_training(params_in: RuntimeIn, params_out: RuntimeOut):

    default_args = {
        'batch_size': 128,
        'epochs': 2,
        'lr': .01,
        'use_gpu': True
    }    


    # No distributed training in this one
    params_out.activate(rank=0, local_rank=0)

    # log top level process
    log = params_out.log.console
    log.begin(f'Running benchmark em_denoise on training mode')

    # parse input arguments against default ones 
    with log.subproc('Parsing input arguments'):
        args = params_in.bench_args.try_get_dict(default_args=default_args)

    # decide which device to use
    if args['use_gpu'] and torch.cuda.is_available():
        device = "cuda:0"
        log.message('Using GPU')
    else:
        device = "cpu"
        log.message('Using CPU')

    # save parameters
    args_file = params_in.output_dir / 'training_arguments_used.yml'
    with log.subproc('Saving arguments to a file'):
        with open(args_file, 'w') as handle:
            yaml.dump(args, handle)

    # create datasets
    with log.subproc('Loading datasets'):
        train_dataset_loader = get_train_data_generator(params_in.dataset_dir / 'train', args["batch_size"])

    # create model
    with log.subproc('Creating the model'):
        model = EMDenoiseNet(input_shape=(256, 256, 1)).to(device)

    # train the model
    with log.subproc('Training the model'):
        params_out.system.stamp_event('start training')
        history = train_model(log, model, train_dataset_loader, args, device)

    # save model
    with log.subproc('Saving (entire) model to a file'):
        model_file = params_in.output_dir / f'em_denoise_model.h5'
        torch.save(model, model_file)

    # save history
    with log.subproc('Saving training history'):
        history_file = params_in.output_dir / 'training_history.yml'
        with open(history_file, 'w') as handle:
            yaml.dump(history, handle)

    # end top level
    log.ended(f'Running benchmark em_denoise on training mode')
    



#####################################################################
# Inference mode                                                    #
#####################################################################

def sciml_bench_inference(params_in: RuntimeIn, params_out: RuntimeOut):

    default_args = {
        'use_gpu': True
    }

    params_out.activate(rank=0, local_rank=0)

    log = params_out.log

    log.begin('Running benchmark em_denoise on inference mode')

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

    log.ended('Running benchmark em_denoise on inference mode')
