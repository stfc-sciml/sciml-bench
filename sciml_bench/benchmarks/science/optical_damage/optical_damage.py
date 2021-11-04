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
from sklearn.metrics import accuracy_score, f1_score
from sciml_bench.core.tensorflow import LogEpochCallback
from sciml_bench.core.utils import MultiLevelLogger

from tqdm import tqdm
from pathlib import Path

import gc
import resource

from sciml_bench.benchmarks.science.optical_damage.model import autoencoder
from sciml_bench.benchmarks.science.optical_damage.utils import IMAGE_SHAPE, load_images

import time
import yaml
from pathlib import Path

from sciml_bench.core.runtime import RuntimeIn, RuntimeOut

def set_target_devices(use_gpu: bool, log: MultiLevelLogger) -> bool:
    if not use_gpu:
        tf.config.set_visible_devices([], 'GPU')
        log.message('Using CPU')
    else:
        log.message('Using GPU')

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
        'batch_size': 64,
        'epochs': 5,
        'latent_size': 512,
        'lr': .001,
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
    set_target_devices(args['use_gpu'], log)

    basePath = params_in.dataset_dir
    modelPath = params_in.output_dir / f'opticsModel.h5'
   
    # Load training images
    with log.subproc('Loading training data'):
        training_path = basePath / 'training/undamaged'
        training_images = load_images(training_path)

        log.message(f'Training images: {len(training_images)}')

    # Load validation images
    with log.subproc('Loading validation data'):
        validation_path = basePath / 'validation/undamaged'
        validation_images = load_images(validation_path)

        log.message(f'Validation images: {len(validation_images)}')


    latent_dim = args['latent_size']
    learning_rate = args['lr']
    batch_size = args['batch_size']

    mirrored_strategy = tf.distribute.MirroredStrategy()

    train_data = tf.data.Dataset.from_tensor_slices((training_images, training_images))
    train_data = train_data.shuffle(2000)
    train_data = train_data.batch(batch_size * mirrored_strategy.num_replicas_in_sync)

    validation_data = tf.data.Dataset.from_tensor_slices((validation_images, validation_images))
    validation_data = validation_data.batch(batch_size * mirrored_strategy.num_replicas_in_sync)

    # Scale the model to multiple GPUs if possible
    with mirrored_strategy.scope():
        opt = tf.keras.optimizers.Adam(learning_rate)
        model = autoencoder(IMAGE_SHAPE, latent_dim)
        model.compile(optimizer=opt, loss='mse', metrics=['mae', 'mse'])

    with log.subproc('Fitting model parameters'):
        history = model.fit(train_data, validation_data=validation_data, epochs=args['epochs'], 
                            verbose=0, callbacks=[LogEpochCallback(params_out)])

    # Free up memory 
    del training_images
    del validation_images
    gc.collect()

    # Save model
    with log.subproc('Saving model file'):
        tf.keras.models.save_model(model, modelPath)
        log.message(f'model saved in:{str(modelPath)}')

    # Peak memory usage
    memory_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    log.message(f'memory_usage in Gbytes: {(memory_usage/(1024*1024)):.2f}')

    # Save history
    with log.subproc('Saving training history'):
        history_file = params_in.output_dir / 'training_history.yml'
        with open(history_file, 'w') as handle:
            yaml.dump(history.history, handle)

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
    set_target_devices(args['use_gpu'], log)

    # Save inference parameters
    args_file = params_in.output_dir / 'inference_arguments_used.yml'
    with log.subproc('Saving inference arguments to a file'):
        with open(args_file, 'w') as handle:
            yaml.dump(args, handle)  
    
    # Reading model path from the command line 
    modelPath = params_in.model

     # Loading the model
    model = tf.keras.models.load_model(modelPath)
    basePath = params_in.dataset_dir

    # The inference folder contains: 4499 undamaged and 2047 damaged images
    with log.subproc('Load damaged images'):
        damaged_path = basePath / 'inference/damaged'
        damaged_images = load_images(damaged_path)
        log.message(f'Number of damaged images: {len(damaged_images)}')
  
    with log.subproc('Load undamaged images'):
        undamaged_path = basePath / 'inference/undamaged'
        undamaged_images = load_images(undamaged_path)
        log.message(f'Number of damaged images: {len(undamaged_images)}')

    # Concatenate two arrays
    test_images = np.concatenate((damaged_images, undamaged_images))

    # Make labels for each image
    test_labels = np.zeros((len(test_images)))
    test_labels[:len(damaged_images)] = 1

    log.message(f'Total number of images use for inferencing: {len(test_images)}') 

    # Inference
    recons = []
    imgs = []
    mse = []

    startInference = time.time()
    with log.subproc('Start inference'):
        for images in tqdm(test_images):
            images = images.reshape(-1, 200, 200, 1)
            recon = model(images, training=False)
            loss = tf.keras.losses.mean_squared_error(images, recon)
            recons.append(recon.numpy())
            imgs.append(images)
            mse.append(loss)

    endInference = time.time()
    log.message(f'Inference time: {(endInference - startInference): 9.4f}, ' \
                f'Images/s: {len(test_images)/(endInference - startInference):9.1f}')

    # Calculating stats
    recons = np.concatenate(recons, axis=0)
    recons = np.squeeze(recons)

    imgs = np.concatenate(imgs, axis=0)
    imgs = np.squeeze(imgs)

    mse = np.concatenate(mse, axis=0)
    mse = np.squeeze(mse)

    with log.subproc('Calculating inference statistics'):
        score_999 = np.any(mse > np.percentile(mse, 99.9), axis=(1, 2))
        score_995 = np.any(mse > np.percentile(mse, 99.5), axis=(1, 2))
        score_990 = np.any(mse > np.percentile(mse, 99.0), axis=(1, 2))

        acc_999 = accuracy_score(score_999, test_labels)
        acc_995 = accuracy_score(score_995, test_labels)
        acc_990 = accuracy_score(score_990, test_labels)

        f1_999 = f1_score(score_999, test_labels)
        f1_995 = f1_score(score_995, test_labels)
        f1_990 = f1_score(score_990, test_labels)

        log.message(f'Accuracy at 99.0%: {acc_990: .2f}, 99.5%: {acc_995:.2f}, 99.9%: {acc_999:.2f}')
        log.message(f'F1 Score at 99.0%: {f1_990: .2f}, 99.5%: {f1_995:.2f}, 99.9%: {f1_999:.2f}')

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
