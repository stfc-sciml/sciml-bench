#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# MNIST_tf_keras.py

# SciML-Bench
# Copyright © 2021 Scientific Machine Learning Research Group
# Scientific Computing Department, Rutherford Appleton Laboratory
# Science and Technology Facilities Council, UK. 
# All rights reserved.

"""
Benchmark: MNIST_tf_keras
Classifying MNIST using a CNN implemented with tf.keras

NOTES:
* This is an example of how to build a benchmark into SciML-Benchmarks.
* It is registered in registration.yml as MNIST_tf_keras.
* In this example, we put everything in MNIST_tf_keras.py; for a real
  problem, it is always better to modularize the implementation, e.g.,
  see MNIST_torch.py; only `sciml_bench_run()` must be implemented here.
"""

# libs from sciml_bench
from sciml_bench.core.runtime import RuntimeIn, RuntimeOut

# libs required by implementation
import time
import h5py
import yaml
import numpy as np
import tensorflow as tf


def create_dataset_mnist(file_path, batch_size):
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

    # create dataset
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

    def __init__(self, smlb_out):
        super().__init__()
        self._start_time = time.time()
        self._smlb_out = smlb_out

    def on_epoch_begin(self, epoch, logs=None):
        # stamp epoch in system monitor
        self._start_time = time.time()
        self._smlb_out.system.stamp_event(f'epoch {epoch}')

    def on_epoch_end(self, epoch, logs=None):
        msg = f'Epoch {epoch:2d}: '
        for key, val in logs.items():
            msg += f'{key}={val:f} '
        msg += f'elapsed={time.time() - self._start_time:f} sec'
        self._smlb_out.log.message(msg)


def sciml_bench_run(smlb_in: RuntimeIn, smlb_out: RuntimeOut):
    """
    Main entry of `sciml_bench run` for a benchmark instance

    :param smlb_in: runtime input of `sciml_bench run`, useful components:
        * smlb_in.start_time: start time of running as UTC-datetime
        * smlb_in.dataset_dir: dataset directory
        * smlb_in.output_dir: output directory
        * smlb_in.bench_args: benchmark-specific arguments
    :param smlb_out: runtime output of `sciml_bench run`, useful components:
        * smlb_out.log.console: multi-level logger on root (rank=0)
        * smlb_out.log.host: multi-level logger on host (local_rank=0)
        * smlb_out.log.device: multi-level logger on device (rank=any)
        * smlb_out.system: a set of system monitors
    """
    # activate monitor
    # Note: To use smlb_out, you must activate it, passing the rank
    #       information initialized by your distributed learning environment;
    #       for a non-distributed benchmark, simply pass rank=0, local_rank=0
    #       and activate_log_on_host(_device)=False; here we use True for
    #       demonstration -- the log on host0 and device0 will be the same as
    #       that on console except for some small differences in time
    #       measurements.
    smlb_out.activate(rank=0, local_rank=0, activate_log_on_host=True,
                      activate_log_on_device=True, console_on_screen=True)

    # log top level process
    # Note: Calling begin(), ended() and message() on smlb_out.log means
    #       calling these functions on console, host and device; nothing
    #       happens when calling these functions on an unactivated logger.
    log = smlb_out.log
    log.begin('Running benchmark MNIST_tf_keras')

    # parse input arguments (only batch_size and epochs)
    # Note: Use try_get() to get a benchmark-specific argument safely from
    #       smlb_in.bench_args (passed by users via -b).
    with log.subproc('Parsing input arguments'):
        # hyperparameters
        batch_size = smlb_in.bench_args.try_get('batch_size', default=64)
        epochs = smlb_in.bench_args.try_get('epochs', default=2)
        log.message(f'batch_size = {batch_size}')
        log.message(f'epochs     = {epochs}')

    # create datasets
    with log.subproc('Creating datasets'):
        dataset_dir = smlb_in.dataset_dir
        train_set = create_dataset_mnist(dataset_dir / 'train.hdf5', batch_size)
        test_set = create_dataset_mnist(dataset_dir / 'test.hdf5', batch_size)
        log.message(f'Dataset directory: {dataset_dir}')

    # create model
    with log.subproc('Creating CNN model'):
        model = create_model_mnist()

    # train model
    log.begin('Training CNN model')
    # fit()
    with log.subproc('Running model.fit()'):
        # stamp model.fit in system monitor
        # Note: smlb_out.system will monitor system usage regularly; use
        #       smlb_out.system.stamp_event() to stamp an event in the report
        smlb_out.system.stamp_event('model.fit')
        history = model.fit(train_set, epochs=epochs, batch_size=batch_size,
                            validation_data=test_set, verbose=0,
                            callbacks=[LogEpochCallback(smlb_out)])
    # save model
    with log.subproc('Saving model weights'):
        weights_file = smlb_in.output_dir / 'model_weights.h5'
        model.save(weights_file)
        log.message(f'Saved to: {weights_file}')
    # save history
    with log.subproc('Saving training history'):
        history_file = smlb_in.output_dir / 'training_history.yml'
        with open(history_file, 'w') as handle:
            yaml.dump(history.history, handle)
        log.message(f'Saved to: {history_file}')
    log.ended('Training CNN model')

    # predict
    with log.subproc('Making predictions on test set'):
        with h5py.File(dataset_dir / 'test.hdf5', 'r') as h5_file:
            # stamp model.predict in system monitor
            smlb_out.system.stamp_event('model.predict')
            pred = model.predict(np.expand_dims(h5_file['image'][:], -1) / 255)
            correct = np.sum(pred.argmax(axis=1) == h5_file['label'][:])
        log.message(f'{correct} correct predictions for {len(pred)} images '
                    f'(accuracy: {correct / len(pred) * 100:.2f}%)')

    # end top level
    log.ended('Running benchmark MNIST_tf_keras')
