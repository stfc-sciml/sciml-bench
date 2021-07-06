#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# em_denoise.py

# SciML-Bench
# Copyright © 2021 Scientific Machine Learning Research Group
# Scientific Computing Department, Rutherford Appleton Laboratory
# Science and Technology Facilities Council, UK. 
# All rights reserved.

"""
Benchmark: em_denoise
Denoising electron microscopy (EM) images of graphene using an autoencoder,
implemented with Apache MXNet.
"""

import numpy as np
import h5py
import yaml
import matplotlib.pyplot as plt

import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
import mxnet.ndarray as nd
from mxnet import autograd as ag
from mxnet.gluon.utils import split_and_load

from sciml_bench.core.runtime import RuntimeIn, RuntimeOut
from sciml_bench.benchmarks.em_denoise.em_denoise_net import EMDenoiseNet
from sciml_bench.benchmarks.em_denoise.em_denoise_util import *
from sciml_bench.core.utils import list_all_files_in_dir



def preprocess(x, y, decimation):
    """ Preprocess images"""
    # channel is last in dataset
    x = mx.nd.moveaxis(x, 3, 1)
    y = mx.nd.moveaxis(y, 3, 1)
    # decimate
    if decimation > 1:
        x = x[:, :, ::decimation, ::decimation]
        y = y[:, :, ::decimation, ::decimation]
    # normalize to [0, 1]
    x = (x - nd.min(x)) / (nd.max(x) - nd.min(x))
    y = (y - nd.min(y)) / (nd.max(y) - nd.min(y))
    return x, y

def train_model(log: MultiLevelLogger, ctx, args, train_iter, test_iter, trainer, model):

    # use MSE as loss function
    loss_func = gluon.loss.L2Loss()
    # history
    history = {'train_loss': [], 'validate_loss': []}

    for epoch in range(args["epochs"]):
        train_loss_last_batch = 0.
        train_iter.reset()
        for i_batch, batch in enumerate(train_iter):
            # load data batch
            noise = split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
            clean = split_and_load(batch.data[1], ctx_list=ctx, batch_axis=0)
            # train
            with ag.record():
                for x, y in zip(noise, clean):
                    x, y = preprocess(x, y, args["decimation"])
                    z = model(x)
                    loss = loss_func(z, y)
                    loss.backward()
                    train_loss_last_batch = loss.mean().asscalar()
            trainer.step(batch.data[0].shape[0])
        history['train_loss'].append(train_loss_last_batch)

        # validate
        val_loss_batch = []
        test_iter.reset()
        for i_batch, batch in enumerate(test_iter):
            # load data batch
            noise = split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
            clean = split_and_load(batch.data[1], ctx_list=ctx, batch_axis=0)
            for x, y in zip(noise, clean):
                x, y = preprocess(x, y, args["decimation"])
                z = model(x)
                val_loss_batch.append(loss_func(z, y).mean().asscalar())
        history['validate_loss'].append(
            nd.array(val_loss_batch).mean().asscalar())

        # message
        log.message(f'Epoch {epoch + 1:3d}: '
                    f'train_loss={history["train_loss"][epoch]:f}, '
                    f'validate_loss={history["validate_loss"][epoch]:f}, '
                    f'elapsed={log.elapsed_shallowest:f} sec')

    return history


#####################################################################
# Training mode                                                     #
#####################################################################

def sciml_bench_training(params_in: RuntimeIn, params_out: RuntimeOut):
    # No distributed training in this one
    params_out.activate(rank=0, local_rank=0)

    # log top level process
    log = params_out.log.console
    log.begin(f'Running benchmark em_denoise on training mode')

    # parse input arguments against default ones 
    with log.subproc('Parsing input arguments'):
        args = parse_training_arguments(params_in.bench_args)

    # save parameters
    args_file = params_in.output_dir / 'training_arguments_used.yml'
    with log.subproc('Saving arguments to a file'):
        with open(args_file, 'w') as handle:
            yaml.dump(args, handle)

    # set the device
    ctx, message = setup_ctx(args)
    log.message(message)

    # create datasets
    with log.subproc('Loading datasets'):
        train_iter, test_iter = create_emdenoise_datasets(params_in.dataset_dir, args["batch_size"])

    # create model
    with log.subproc('Creating the model'):
        model, trainer = create_em_denoise_model(args, ctx, xavier_mag=2.24)

    # train the model
    with log.subproc('Training the model'):
        params_out.system.stamp_event('start training')
        history = train_model(log, ctx, args, train_iter, test_iter, trainer, model)

    # save model
    with log.subproc('Saving model to a file'):
        model_file = params_in.output_dir / f'em_denoise_model_decimation_{args["decimation"]}.h5'
        model.save_parameters(str(model_file))

    # save history
    with log.subproc('Saving training history'):
        history = organise_history(history)
        history_file = params_in.output_dir / 'training_history.yml'
        with open(history_file, 'w') as handle:
            yaml.dump(history, handle)

    # end top level
    log.ended(f'Running benchmark em_denoise on training mode')




#####################################################################
# Inference mode                                                    #
#####################################################################


def sciml_bench_inference(params_in: RuntimeIn, params_out: RuntimeOut):
    pass

  
    # log activation , etc
    params_out.activate(rank=0, local_rank=0)
    log = params_out.log

    log.begin('Running benchmark em_denoise on inference mode')

    # parse input arguments against default ones 
    with log.subproc('Parsing input arguments'):
        args = parse_inference_arguments(params_in.bench_args)

    # save parameters
    args_file = params_in.output_dir / 'inference_arguments_used.yml'
    with log.subproc('Saving arguments to a file'):
        with open(args_file, 'w') as handle:
            yaml.dump(args, handle)

    # set the device
    ctx, message = setup_ctx(args)
    log.message(message)

    with log.subproc('Loading model'):
        model_file = load_model(params_in.model)

    images = [plt.read(x) for x in list_all_files_in_dir(params_in.dataset_dir)]
    batch = nd.array(np.concatenate([transform(img) for img in images], axis=0), ctx=ctx)
    
    with log.subproc('Performing inference on data files'):
        inference_outputs = run_batch(model, inference_inputs)
    
    # save the inference outputs
    # todo

    log.ended('Running benchmark em_denoise on inference mode')