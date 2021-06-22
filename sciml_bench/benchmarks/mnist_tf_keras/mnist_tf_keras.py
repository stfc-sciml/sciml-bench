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


NOTES:
* This is an example of how to build a benchmark into SciML-Bench.
* Please see the configuration options in config.yml.
* Although this example relies on single file implementation,
* in reality, it is always better to modularize the implementation.
  As this benchmark supports both training and inference,
  both  `sciml_bench_training()` and `sciml_bench_inference()`
  must be implemented here.
"""

# libs from sciml_bench
from sciml_bench.core.runtime import RuntimeIn, RuntimeOut

# libs required by implementation
import time
import h5py
import yaml
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tf.keras.preprocessing.image import load_img
from tf.keras.preprocessing.image import img_to_array
from pathlib import Path


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
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                                     input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    # compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


class LogEpochCallback(tf.keras.callbacks.Callback):
    """ Callback to log epoch """

    def __init__(self, params_out):
        super().__init__()
        self._start_time = time.time()
        self._params_out = params_out

    def on_epoch_begin(self, epoch, logs=None):
        # stamp epoch in system monitor
        self._start_time = time.time()
        self._params_out.system.stamp_event(f'epoch {epoch}')

    def on_epoch_end(self, epoch, logs=None):
        msg = f'Epoch {epoch:2d}: '
        for key, val in logs.items():
            msg += f'{key}={val:f} '
        msg += f'elapsed={time.time() - self._start_time:f} sec'
        self._params_out.log.message(msg)


def sciml_bench_training(params_in: RuntimeIn, params_out: RuntimeOut):
    """
    Main entry of `sciml_bench run` for a benchmark instance
    in the training mode.
    Please consult the API documentation. 
    """
    
    params_out.activate(rank=0, local_rank=0, activate_log_on_host=True,
                      activate_log_on_device=True, console_on_screen=True)

    log = params_out.log
    log.begin('Running benchmark mnist_tf_keras on training mode')

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
        model_file = params_in.output_dir / 'mnist_tf_keras_model.h5'
        model.save(model_file)
        log.message(f'Saved to: {model_file}')
    # save history
    with log.subproc('Saving training history'):
        history_file = params_in.output_dir / 'training_history.yml'
        with open(history_file, 'w') as handle:
            yaml.dump(history.history, handle)
        log.message(f'Saved to: {history_file}')
    log.ended('Training CNN model')

    # predict
    with log.subproc('Making predictions on test set'):
        with h5py.File(dataset_dir / 'test.hdf5', 'r') as h5_file:
            # stamp model.predict in system monitor
            params_out.system.stamp_event('model.predict')
            pred = model.predict(np.expand_dims(h5_file['image'][:], -1) / 255)
            correct = np.sum(pred.argmax(axis=1) == h5_file['label'][:])
        log.message(f'{correct} correct predictions for {len(pred)} images '
                    f'(accuracy: {correct / len(pred) * 100:.2f}%)')

    # end top level
    log.ended('Running benchmark mnist_tf_keras on training mode')



def sciml_bench_inference(params_in: RuntimeIn, params_out: RuntimeOut):
    """
    Main entry of `sciml_bench run` for a benchmark instance 
    in the inference mode.
    Please consult the API documentation. 
    """
    params_out.activate(rank=0, local_rank=0, activate_log_on_host=True,
                      activate_log_on_device=True, console_on_screen=True)

    log = params_out.log
    log.begin('Running benchmark mnist_tf_keras on inference mode')

    output_file = params_in.output_dir  / 'mnist_tf_keras_inference.log'

    # Load the model
    with log.subproc('Model loading and inference'):
        # There is only one model we have - named tf_mnist_keras_model
        model = load_model(params_in.models['mnist_tf_keras_model'])
        # Process each file 
        p = params_in.dataset_dir.glob('**/*')
        files = [x for x in p if x.is_file()]
        with open(output_file, 'wt')as handle:
            handle.write(f'File Name\tOutput\n')
            for file in sorted(files):
                with log.subproc(f'Processing file {file}'):
                    image = load_img(file, color_mode='grayscale', target_size=(28,28))
                    input_arr = img_to_array(image)/ 255.0
                    input_arr = np.array([input_arr]).reshape(-1, 28, 28, 1) 
                    out = model.predict(input_arr)
                    out = np.argmax(out)
                    handle.write(f'{file}\t{out}\n')

    log.ended('Running benchmark mnist_tf_keras on inference mode')