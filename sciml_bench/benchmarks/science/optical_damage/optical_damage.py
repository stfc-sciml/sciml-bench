#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# optical_damage.py

# SciML-Bench
# Copyright © 2021 Scientific Machine Learning Research Group
# Scientific Computing Department, Rutherford Appleton Laboratory
# Science and Technology Facilities Council, UK. 
# All rights reserved.

import h5py
import numpy as np
import tensorflow as tf
from pathlib import Path

import PIL
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import sys
from sys import getsizeof
import glob
from matplotlib import pyplot

import gc
import resource

# Import model and methods
from sciml_bench.benchmarks.science.optical_damage.opticalDamageModel import autoencoder
from sciml_bench.benchmarks.science.optical_damage.opticalDamageUtils import normalize
from sciml_bench.benchmarks.science.optical_damage.opticalDamageUtils import _crop_bb
from sciml_bench.benchmarks.science.optical_damage.opticalDamageUtils import processImages
from sciml_bench.benchmarks.science.optical_damage.opticalDamageUtils import ssim_loss

import time
import yaml
import torch
import math
from pathlib import Path

from sciml_bench.core.runtime import RuntimeIn, RuntimeOut

#####################################################################
# Training mode                                                     #
#####################################################################
# For training use this command:
# sciml-bench run --mode training optical_damage

def sciml_bench_training(params_in: RuntimeIn, params_out: RuntimeOut):
    """
    Entry point for the training routine to be called by SciML-Bench
    """
    default_args = {
        'batch_size': 128,
        'epochs': 1,
        'lr': .01,
        'use_gpu': True
    }    

    # No distributed training in this one
    params_out.activate(rank=0, local_rank=0)

    # Log top level process
    log = params_out.log.console
    log.begin(f'Running benchmark optical_damage on training mode')

    # Parse input arguments against default ones 
    with log.subproc('Parsing input arguments'):
        args = params_in.bench_args.try_get_dict(default_args=default_args)
    
    # Save parameters
    args_file = params_in.output_dir / 'training_arguments_used.yml'
    with log.subproc('Saving arguments to a file'):
        with open(args_file, 'w') as handle:
            yaml.dump(args, handle)

    # Decide which device to use
    if args['use_gpu'] and torch.cuda.is_available():
        device = "cuda:0"
        log.message('Using GPU')
    else:
        device = "cpu"
        log.message('Using CPU')

    basePath = params_in.dataset_dir
    # Save all models in one dir
    modelPath = params_in.output_dir / f'opticsModel.h5'
   
    # Copy processed files into new dir
    source = basePath / 'training/undamaged'
    destination = basePath / 'training/undamagedProcessed'

     # check if directory exists, if not ctreate one
    Path(destination).mkdir(parents=True, exist_ok=True)

    numberOfProcessedImages = processImages(source, destination, 'undamaged')
    log.message(f'Training images: {len(numberOfProcessedImages)}')

    # Process validation images
    source = basePath / 'validation/undamaged'
    destination = basePath / 'validation/undamagedProcessed'

    # check if directory exists, if not ctreate one
    Path(destination).mkdir(parents=True, exist_ok=True)

    numberOfValidImages = processImages(source, destination, 'undamaged')
    log.message(f'Validation images: {len(numberOfValidImages)}')

    #load processed images for model input
    inputImages = []
    for item in numberOfProcessedImages:
        loadedImage = np.load(item)
        inputImages.append(loadedImage)
    inputImages = np.array(inputImages)

    # Load validation images
    validImages = []
    for item in numberOfValidImages:
        loadedImage = np.load(item)
        validImages.append(loadedImage)
    validImages = np.array(validImages)

    opt = tf.keras.optimizers.Adam(0.001)
    input_shape=(200, 200, 1)
    latent_dim=2000
    model1 = autoencoder(input_shape, latent_dim)

    # Metrics Mean Absolute Error (mae)
    model1.compile(optimizer=opt, loss='mse', metrics=['mae', 'mse'])
    with log.subproc('Calculating model parameters'):
        history1 = model1.fit(inputImages, inputImages, batch_size=32, epochs=1, verbose=2)

    # Free up memory 
    del inputImages
    del validImages
    gc.collect()

    # Save model
    with log.subproc('Saving model file'):
        tf.keras.models.save_model(model1, modelPath)
        log.message(f'model saved in:{str(modelPath)}')

    # Peak memory usage
    memory_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    log.message(f'memory_usage in Gbytes: {(memory_usage/(1024*1024)):.2f}')

    # Save history
    with log.subproc('Saving training history'):
        history_file = params_in.output_dir / 'training_history.yml'
        with open(history_file, 'w') as handle:
            yaml.dump(history1.history, handle)

    # End top level
    log.ended(f'Running benchmark optical_damage on training mode')
    

#####################################################################
# Inference mode                                                    #
#####################################################################
# For inference use this command:
# sciml-bench run --mode inference --model ~/sciml_bench/models/optical_damage/opticsModel.h5 --dataset_dir ~/sciml_bench/datasets/optical_damage_ds1 optical_damage

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

    log.begin('Running benchmark optical_damage on inference mode')

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
    
    # Reading model path from the command line 
    modelPath = params_in.model

     # Loading the model
    model = tf.keras.models.load_model(modelPath, custom_objects={'ssim_loss': ssim_loss})
    basePath = params_in.dataset_dir

    # The inference folder contains: 4499 undamaged and 2047 damaged images
    # Copy damaged images into new dir
    source = basePath / 'inference/damaged'
    destination = basePath / 'inference/processedImages'
    
    # Check if directory exists, if not create one
    Path(destination).mkdir(parents=True, exist_ok=True)
    with log.subproc('Process damaged images'):
        damagedList = processImages(source, destination, 'damaged')
        log.message(f'Number of damaged images: {len(damagedList)}')
  
    # Copy undamaged images into new dir
    source = basePath / 'inference/undamaged'
    with log.subproc('Process undamaged images'):
        undamagedList = processImages(source, destination, 'undamaged')
        log.message(f'Number of damaged images: {len(undamagedList)}')

    # Concatenate two arrays
    allProcessedTestImages = np.concatenate((damagedList, undamagedList))
    # log.message(f'Total number of images use for inferencing: {len(allProcessedTestImages)}') 

    # Load test images for inference 
    with log.subproc('Loading inference images'):
        #log.message(f'Loading inference images')
        testImages = []
        for item in allProcessedTestImages:
            loadedImage = np.load(item)
            testImages.append(loadedImage)
        testImages = np.array(testImages)
        # print(f'testImages_size: {len(testImages)}') 
        log.message(f'Inference images loaded: {len(testImages)}')

    # Inference
    recons = []
    imgs = []
    mse = []
    startInference = time.time()
    with log.subproc('Start inference'):
        for images in tqdm(testImages):
            images = images.reshape(-1, 200, 200, 1)
            recon = model(images, training=False)
            loss = np.square(np.subtract(images, recon)).mean()
            recons.append(recon.numpy())
            imgs.append(images)
            mse.append(loss)
        log.message(f'mse_lenght: {len(mse)}')

    endInference = time.time()
    log.message(f'Inference time: {(endInference - startInference): 9.4f}, \
            Images/s: {len(testImages)/(endInference - startInference):9.1f}')

    # MSE distribution plot
    # sns.histplot(data=mse, kde=True, stat="probability")
    # plt.show()

    # Calculating stats
    recons = np.concatenate(recons, axis=0)
    recons = np.squeeze(recons)
    imgs = np.concatenate(imgs, axis=0)
    imgs = np.squeeze(imgs)

    # mse is a zero dimensional array, cannot be concatenated
    # mse = np.concatenate(mse, axis=0)
    # mse = np.squeeze(mse)

    # Max memory usage
    memory_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    log.message(f'memory_usage in Gbytes: {memory_usage/(1024*1024): 9.4f}')

    # Saving images, reconstructions and mse
    with log.subproc('Saving images, reconstructions and mse'):
        output_file = params_in.output_dir / 'inference.h5'
        with h5py.File(output_file, 'w') as handle:
            handle.create_dataset('reconstruction', data=recons)
            handle.create_dataset('images', data=imgs)
            handle.create_dataset('mse', data=mse)
        log.message(f"Reconstructions saved to: {output_file}")

    # End top level
    log.ended('Running benchmark optical_damage on inference mode')
