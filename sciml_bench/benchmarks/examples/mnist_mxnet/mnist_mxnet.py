#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# mnist_tf_keras.py

# SciML-Bench
# Copyright © 2021 Scientific Machine Learning Research Group
# Scientific Computing Department, Rutherford Appleton Laboratory
# Science and Technology Facilities Council, UK. 
# All rights reserved.

"""
Benchmark: mnist_tf_keras
Classifying MNIST using a CNN implemented with tf.keras
This is a single device training/inference example.
"""

# libs from sciml_bench
from sciml_bench.core.runtime import RuntimeIn, RuntimeOut
from sciml_bench.core.utils import list_files

import time
import h5py
import yaml
import numpy as np
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
import mxnet.ndarray as nd
from mxnet import autograd as ag
from mxnet.gluon.utils import split_and_load
from pathlib import Path
import skimage.io


# Some utility codes

def load_dataset_mnist(file_path, batch_size):
    """ Create dataset """

    # generator
    def hdf5_generator(path, batch):
        with h5py.File(path, 'r') as h5_file:
            for i in range(0, h5_file['image'].shape[0], batch):
                # read, expand channel dim and normalize
                images = np.expand_dims(h5_file['image'][i:i + batch], -1) / 255
                # read and one-hot encoding
                labels = np.eye(10)[h5_file['label'][i:i + batch]]
                yield images, labels

    # load dataset
    dataset = tf.data.Dataset.from_generator(
        lambda: hdf5_generator(file_path, batch_size),
        output_types=(tf.float32, tf.float32),
        output_shapes=((None, 28, 28, 1), (None, 10)))
    return dataset


def create_model_mnist():
    """ Create model """
    # create model
    # model = tf.keras.Sequential()
    # model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
    #                                  input_shape=(28, 28, 1)))
    # model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    # model.add(tf.keras.layers.Flatten())
    # model.add(tf.keras.layers.Dense(128, activation='relu'))
    # model.add(tf.keras.layers.Dense(10, activation='softmax'))

    # compile model
    # model.compile(optimizer='adam', loss='categorical_crossentropy',
                #   metrics=['accuracy'])

    model = None
    return model


def load_images(image_dir_path):
    file_names =  list_files(image_dir_path)
    images = np.zeros((len(file_names), 28, 28, 1))
    for idx, url in enumerate(file_names):
        images[idx, :, :, 0] = skimage.io.imread(url)
    return images, file_names



# Training Code

def sciml_bench_training(params_in: RuntimeIn, params_out: RuntimeOut):
    """
    Main entry of `sciml_bench run` for a benchmark instance
    in the training mode.
    Please consult the API documentation. 
    """
    
    params_out.activate(rank=0, local_rank=0, activate_log_on_host=True,
                      activate_log_on_device=True, console_on_screen=True)

    log = params_out.log
    log.begin('Running benchmark mnist_mxnet on training mode')

    # We expect two benchmark-specific arguments here: 
    # batch_size and epochs. If not, we will assign 
    # default values.
    with log.subproc('Parsing input arguments'):
        # hyperparameters
        suggested_args = {
            'batch_size': 128,
            'epochs': 2
        } 
        args = params_in.bench_args.try_get_dict(default_args=suggested_args)
        batch_size = args['batch_size']
        epochs = args['epochs']
        log.message(f'batch_size = {batch_size}')
        log.message(f'epochs     = {epochs}')

    with log.subproc('Writing the argument file'):
        args_file = params_in.output_dir / 'arguments_used.yml'
        with open(args_file, 'w') as handle:
            yaml.dump(args, handle)

    # create datasets
    with log.subproc('Creating datasets'):
        dataset_dir = params_in.dataset_dir
        train_set = load_dataset_mnist(dataset_dir / 'train.hdf5', batch_size)
        test_set = load_dataset_mnist(dataset_dir / 'test.hdf5', batch_size)
        log.message(f'Dataset directory: {dataset_dir}')

    # create model
    with log.subproc('Creating CNN model'):
        model = create_model_mnist()

    # train model
    log.begin('Training CNN model')
    with log.subproc('Running model.fit()'):
        params_out.system.stamp_event('model.fit')
        history = model.fit(train_set, epochs=epochs, batch_size=batch_size,
                            validation_data=test_set, verbose=0,
                            callbacks=[LogEpochCallback(params_out)])
    # save model
    with log.subproc('Saving the model'):
        model_file = params_in.output_dir / 'mnist_mxnet_model.h5'
        model.save(model_file)
        log.message(f'Saved to: {model_file}')
    # save history
    with log.subproc('Saving training history'):
        history_file = params_in.output_dir / 'training_history.yml'
        with open(history_file, 'w') as handle:
            yaml.dump(history.history, handle)
        log.message(f'Saved to: {history_file}')
    log.ended('Training CNN model')


    # end top level
    log.ended('Running benchmark mnist_mxnet on training mode')




# Inference Code

def sciml_bench_inference(params_in: RuntimeIn, params_out: RuntimeOut):

    params_out.activate(rank=0, local_rank=0, activate_log_on_host=True,
                      activate_log_on_device=True, console_on_screen=True)

    log = params_out.log

    log.begin('Running benchmark mnist_mxnet on inference mode')

    # Load the model and perform bulk inference 
    with log.subproc('Model loading and inference'):
        model = load_model(params_in.model)
        raw_images, file_names = load_images(params_in.dataset_dir)
        images = tf.constant(raw_images, dtype=np.float32)
        outputs = model.predict(images)
        mappings = np.argmax(outputs, axis=1)

    # Write out the resultsa
    with log.subproc('Writing to outputs'):
        for i in range(len(file_names)):
            log.message(f'{file_names[i]}\t{mappings[i]}')
    

    log.ended('Running benchmark mnist_mxnet on inference mode')