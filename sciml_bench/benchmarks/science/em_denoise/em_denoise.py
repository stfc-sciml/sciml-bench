#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# em_denoise.py

# SciML-Bench
# Copyright © 2021 Scientific Machine Learning Research Group
# Scientific Computing Department, Rutherford Appleton Laboratory
# Science and Technology Facilities Council, UK. 
# All rights reserved.


import time
import yaml
import torch
import math
from pathlib import Path

from sciml_bench.core.runtime import RuntimeIn, RuntimeOut

from sciml_bench.benchmarks.science.em_denoise.em_denoise_util import EMDenoiseInferenceDataset
from sciml_bench.benchmarks.science.em_denoise.em_denoise_util import EMDenoiseTrainingDataset
from sciml_bench.benchmarks.science.em_denoise.em_denoise_util import train_model
from sciml_bench.benchmarks.science.em_denoise.em_denoise_model import EMDenoiseNet


def get_data_generator(base_dataset_dir: Path, batch_size: int, is_inference=False):
    """
    Returns a data loader for training or inference datasets 
    based on the is_inference flag.
    """
    shuffle_flag = True
    if is_inference:
       shuffle_flag = False

    params = {
        'batch_size': batch_size,
        'shuffle': shuffle_flag,
        'num_workers': 2
    }

    if is_inference: 
        inference_path = base_dataset_dir / 'raw'
        inference_gt_path = base_dataset_dir / 'truth'
        em_inference_dataset = EMDenoiseInferenceDataset(inference_path, inference_gt_path)
        em_inference_generator = torch.utils.data.DataLoader(em_inference_dataset, **params)
        return em_inference_generator
    else: 
        noisy_path = str(base_dataset_dir / 'graphene_img_noise.h5')
        clean_path = str(base_dataset_dir / 'graphene_img_clean.h5')
        em_denoise_dataset = EMDenoiseTrainingDataset(noisy_path, clean_path)
        em_denoise_generator = torch.utils.data.DataLoader(em_denoise_dataset, **params)
        return em_denoise_generator



#####################################################################
# Training mode                                                     #
#####################################################################

def sciml_bench_training(params_in: RuntimeIn, params_out: RuntimeOut):
    """
    Entry point for the training routine to be called by SciML-Bench
    """
    default_args = {
        'batch_size': 128,
        'epochs': 10,
        'lr': .01,
        'use_gpu': True
    }    


    # No distributed training in this one
    params_out.activate(rank=0, local_rank=0)

    # Log top level process
    log = params_out.log.console
    log.begin(f'Running benchmark em_denoise on training mode')

    # Parse input arguments against default ones 
    with log.subproc('Parsing input arguments'):
        args = params_in.bench_args.try_get_dict(default_args=default_args)

    # Decide which device to use
    if args['use_gpu'] and torch.cuda.is_available():
        device = "cuda:0"
        log.message('Using GPU')
    else:
        device = "cpu"
        log.message('Using CPU')

    # Save parameters
    args_file = params_in.output_dir / 'training_arguments_used.yml'
    with log.subproc('Saving arguments to a file'):
        with open(args_file, 'w') as handle:
            yaml.dump(args, handle)

    # Create datasets
    with log.subproc('Loading datasets'):
        train_dataset_loader = get_data_generator(params_in.dataset_dir / 'train', args["batch_size"], is_inference=False)

    # Create model
    with log.subproc('Creating the model'):
        model = EMDenoiseNet(input_shape=(256, 256, 1)).to(device)

    # Train the model
    with log.subproc('Training the model'):
        params_out.system.stamp_event('start training')
        start_time = time.time()
        history = train_model(log, model, train_dataset_loader, args, device)
        end_time = time.time()
        time_taken = end_time - start_time

    # Save model
    with log.subproc('Saving (entire) model to a file'):
        model_file = params_in.output_dir / f'em_denoise_model.h5'
        torch.save(model, model_file)

    # Save history
    with log.subproc('Saving training history'):
        history_file = params_in.output_dir / 'training_history.yml'
        with open(history_file, 'w') as handle:
            yaml.dump(history, handle)

    # Save metrics
    metrics = dict(time=time_taken, loss=history[-1])
    metrics_file = params_in.output_dir / 'metrics.yml'
    with log.subproc('Saving inference metrics to a file'):
        with open(metrics_file, 'w') as handle:
            yaml.dump(metrics, handle)  

    # End top level
    log.ended(f'Running benchmark em_denoise on training mode')
    



#####################################################################
# Inference mode                                                    #
#####################################################################

def sciml_bench_inference(params_in: RuntimeIn, params_out: RuntimeOut):
    """
    Entry point for the inference routine to be called by SciML-Bench
    """
    default_args = {
        'use_gpu': True,
        "batch_size" : 64
    }

    params_out.activate(rank=0, local_rank=0)

    log = params_out.log

    # Lambdas for loss computation
    mse_criteria = torch.nn.MSELoss()
    psnr_loss = lambda mse, L: 20 * math.log10(float(L) / math.sqrt(mse))

    log.begin('Running benchmark em_denoise on inference mode')

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

    # Create datasets
    with log.subproc(f'Setting up a data loader for inference'):
        inference_dataset_loader = get_data_generator(params_in.dataset_dir, args["batch_size"], is_inference=True)

    # Load the model and move it to the right device
    with log.subproc(f'Loading the model for inference into {device}'):
        model = torch.load(params_in.model)
        model.to(device)

    # Perform bulk inference on the target device + collect metrics
    with log.subproc(f'Doing inference across {len(inference_dataset_loader.dataset):,} items on device: {device}'):
        start_time = time.time()
        running_mse = 0.0
        running_psnr = 0.0
        for noisy_batch, clean_batch in inference_dataset_loader:
            model.eval()
            with torch.no_grad():
                noisy_batch = noisy_batch.float().to(device)
                clean_batch = clean_batch.float().to(device)
                denoised_batch = model.forward(noisy_batch)
                batch_mse = mse_criteria(denoised_batch, clean_batch)
                batch_psnr = psnr_loss(batch_mse.item(), L=255)
                running_mse  += batch_mse.item()
                running_psnr += batch_psnr
        mse = running_mse / len(inference_dataset_loader)
        psnr = running_psnr /  len(inference_dataset_loader)
        end_time = time.time()
    time_taken = end_time - start_time

    throughput = math.floor(len (inference_dataset_loader.dataset) / time_taken)

    # Log the outputs
    with log.subproc('Inference Performance'):
        log.message(f'Throughput  : {throughput:,} Images / sec')
        log.message(f'Overall Time: {time_taken:.4f} s')
        log.message(f'Average MSE : {mse:.4f}')
        log.message(f'Average PSNR: {psnr:.4f} dB')

    # Save metrics
    metrics = dict(throughput=throughput, time=time_taken, mse=mse, psnr=psnr)
    metrics_file = params_in.output_dir / 'metrics.yml'
    with log.subproc('Saving inference metrics to a file'):
        with open(metrics_file, 'w') as handle:
            yaml.dump(metrics, handle)  

    # End top level
    log.ended('Running benchmark em_denoise on inference mode')
