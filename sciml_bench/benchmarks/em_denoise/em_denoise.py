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
implemented with Apache MXNet
"""

# libs from sciml_bench
from sciml_bench.core.runtime import RuntimeIn, RuntimeOut

# libs required by implementation
# mxnet
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
import mxnet.ndarray as nd
from mxnet import autograd as ag
from mxnet.gluon.utils import split_and_load
# helpers
import numpy as np
import h5py
import yaml
import matplotlib.pyplot as plt


def create_data_iter(file_noise, file_clean, batch_size):
    """ Create data iterator """
    h5_noise = h5py.File(file_noise, 'r')['images']
    h5_clean = h5py.File(file_clean, 'r')['images']
    return mx.io.NDArrayIter([h5_noise, h5_clean], batch_size=batch_size)


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


class EMDenoiseNet(gluon.Block):
    """ Network class """

    def __init__(self, decimation=1, **kwargs):
        super(EMDenoiseNet, self).__init__(**kwargs)

        def conv2d_3x3_same_padding_relu(channels):
            return nn.Conv2D(channels, kernel_size=(3, 3),
                             padding=(1, 1), activation='relu')

        def conv2d_trans_3x3_same_padding_relu(channels):
            return nn.Conv2DTranspose(channels, kernel_size=(3, 3),
                                      padding=(1, 1), activation='relu')

        base_channels = 8 if decimation <= 2 else 4
        with self.name_scope():
            # block 1
            self.conv11 = conv2d_3x3_same_padding_relu(base_channels)
            self.conv12 = conv2d_3x3_same_padding_relu(base_channels)
            self.bn11 = nn.BatchNorm()
            self.bn12 = nn.BatchNorm()
            self.mp1 = nn.MaxPool2D()
            # block 2
            self.conv21 = conv2d_3x3_same_padding_relu(base_channels * 2)
            self.conv22 = conv2d_3x3_same_padding_relu(base_channels * 2)
            self.bn21 = nn.BatchNorm()
            self.bn22 = nn.BatchNorm()
            self.mp2 = nn.MaxPool2D()
            # block 3
            self.conv31 = conv2d_3x3_same_padding_relu(base_channels * 4)
            self.conv32 = conv2d_3x3_same_padding_relu(base_channels * 4)
            self.bn31 = nn.BatchNorm()
            self.bn32 = nn.BatchNorm()
            self.mp3 = nn.MaxPool2D()
            # block 4
            self.conv41 = conv2d_3x3_same_padding_relu(base_channels * 8)
            self.conv42 = conv2d_3x3_same_padding_relu(base_channels * 8)
            self.bn41 = nn.BatchNorm()
            self.bn42 = nn.BatchNorm()
            # block 5
            self.conv51 = conv2d_trans_3x3_same_padding_relu(base_channels * 4)
            self.conv52 = conv2d_trans_3x3_same_padding_relu(base_channels * 4)
            self.bn51 = nn.BatchNorm()
            self.bn52 = nn.BatchNorm()
            # block 6
            self.conv61 = conv2d_trans_3x3_same_padding_relu(base_channels * 2)
            self.conv62 = conv2d_trans_3x3_same_padding_relu(base_channels * 2)
            self.bn61 = nn.BatchNorm()
            self.bn62 = nn.BatchNorm()
            # block 7
            self.conv71 = conv2d_trans_3x3_same_padding_relu(base_channels)
            self.conv72 = conv2d_trans_3x3_same_padding_relu(base_channels)
            self.bn71 = nn.BatchNorm()
            self.bn72 = nn.BatchNorm()
            # block 8, no activation
            self.conv8 = nn.Conv2D(1, kernel_size=(3, 3), padding=1)

    def forward(self, x):
        # skipped
        skipped = []
        # block 1
        x = self.bn12(self.conv12(self.bn11(self.conv11(x))))
        skipped.append(x)
        x = self.mp1(x)
        # block 2
        x = self.bn22(self.conv22(self.bn21(self.conv21(x))))
        skipped.append(x)
        x = self.mp2(x)
        # block 3
        x = self.bn32(self.conv32(self.bn31(self.conv31(x))))
        skipped.append(x)
        x = self.mp3(x)
        # block 4
        x = self.bn42(self.conv42(self.bn41(self.conv41(x))))
        # block 5
        x = nd.UpSampling(x, scale=2, sample_type='nearest')
        x = nd.concat(x, skipped.pop(-1))
        x = self.bn52(self.conv52(self.bn51(self.conv51(x))))
        # block 6
        x = nd.UpSampling(x, scale=2, sample_type='nearest')
        x = nd.concat(x, skipped.pop(-1))
        x = self.bn62(self.conv62(self.bn61(self.conv61(x))))
        # block 7
        x = nd.UpSampling(x, scale=2, sample_type='nearest')
        x = nd.concat(x, skipped.pop(-1))
        x = self.bn72(self.conv72(self.bn71(self.conv71(x))))
        # block 8
        x = self.conv8(x)
        return x


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
    # No distributed training in this one
    smlb_out.activate(rank=0, local_rank=0)

    # log top level process
    log = smlb_out.log.console
    log.begin('Running benchmark em_denoise')

    # parse input arguments
    with log.subproc('Parsing input arguments'):
        # default arguments
        default_args = {
            # used image size = (256/decimation) x (256/decimation)
            'decimation': 4,
            'batch_size': 128,
            'epochs': 2,
            'lr': .01,
            'use_cuda': True,
            'plot_batches': 1
        }
        # replace default_args with bench_args
        args = smlb_in.bench_args.try_get_dict(default_args=default_args)
        assert args["decimation"] in [1, 2, 4], \
            'Decimation scale must be 1, 2 or 4.'
        # verbose parameters
        log.message(f'decimation   = {args["decimation"]}')
        log.message(f'batch_size   = {args["batch_size"]}')
        log.message(f'epochs       = {args["epochs"]}')
        log.message(f'lr           = {args["lr"]}')
        log.message(f'use_cuda     = {args["use_cuda"]}')
        log.message(f'plot_batches = {args["plot_batches"]}')
        # save parameters
        args_file = smlb_in.output_dir / 'arguments_used.yml'
        with open(args_file, 'w') as handle:
            yaml.dump(args, handle)
        log.message(f'Arguments used are saved to:\n{args_file}')
        # check gpu
        ctx = [mx.cpu()]
        if args["use_cuda"]:
            if mx.test_utils.list_gpus():
                ctx = [mx.gpu()]
                log.message('CUDA is available.')
            else:
                log.message('WARNING: CUDA is unavailable, '
                            'argument "use_cuda=True" is ignored')

    # create datasets
    with log.subproc('Creating datasets'):
        data_dir = smlb_in.dataset_dir
        train_iter = create_data_iter(data_dir / 'train/graphene_img_noise.h5',
                                      data_dir / 'train/graphene_img_clean.h5',
                                      args["batch_size"])
        test_iter = create_data_iter(data_dir / 'test/graphene_img_noise.h5',
                                     data_dir / 'test/graphene_img_clean.h5',
                                     args["batch_size"])
        log.message(f'Dataset directory: {data_dir}')

    # create model
    with log.subproc('Creating autoencoder'):
        net = EMDenoiseNet(decimation=args["decimation"])
        net.initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
        trainer = gluon.Trainer(net.collect_params(), 'adam',
                                {'learning_rate': args["lr"]})

    # train model
    log.begin('Training autoencoder')
    # use MSE as loss function
    loss_func = gluon.loss.L2Loss()
    # history
    history = {'train_loss': [], 'validate_loss': []}

    smlb_out.system.stamp_event('start training')
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
                    # preprocess
                    x, y = preprocess(x, y, args["decimation"])
                    # forward
                    z = net(x)
                    # loss
                    loss = loss_func(z, y)
                    # backward
                    loss.backward()
                    # record
                    train_loss_last_batch = loss.mean().asscalar()
            trainer.step(batch.data[0].shape[0])
        # record loss for last batch
        history['train_loss'].append(train_loss_last_batch)

        # validate
        val_loss_batch = []
        test_iter.reset()
        for i_batch, batch in enumerate(test_iter):
            # load data batch
            noise = split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
            clean = split_and_load(batch.data[1], ctx_list=ctx, batch_axis=0)
            for x, y in zip(noise, clean):
                # preprocess
                x, y = preprocess(x, y, args["decimation"])
                # forward
                z = net(x)
                # loss
                val_loss_batch.append(loss_func(z, y).mean().asscalar())
        # record batch average of loss
        history['validate_loss'].append(
            nd.array(val_loss_batch).mean().asscalar())

        # message
        log.message(f'Epoch {epoch + 1:3d}: '
                    f'train_loss={history["train_loss"][epoch]:f}, '
                    f'validate_loss={history["validate_loss"][epoch]:f}, '
                    f'elapsed={log.elapsed_shallowest:f} sec')
    log.ended('Training autoencoder')

    # save weights and history
    with log.subproc('Saving model weights and training history'):
        # weights
        weights_file = smlb_in.output_dir / \
                       f'model_weights_decimation={args["decimation"]}.params'
        net.save_parameters(str(weights_file))
        log.message(f'Model weights saved to:\n{weights_file}')
        # history
        history['train_loss'] = np.array(history['train_loss']).tolist()
        history['validate_loss'] = np.array(history['validate_loss']).tolist()
        history_file = smlb_in.output_dir / 'training_history.yml'
        with open(history_file, 'w') as handle:
            yaml.dump(history, handle)
        log.message(f'Training history saved to:\n{history_file}')

    # plot batches
    if args["plot_batches"] > 0:
        log.begin('Plotting some batches from test set')
        img_folder = smlb_in.output_dir / 'plots'
        img_folder.mkdir(exist_ok=True)
        plot_count = 0
        test_iter.reset()
        for i_batch, batch in enumerate(test_iter):
            if i_batch >= args["plot_batches"]:
                break
            # load data batch
            noise = split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
            clean = split_and_load(batch.data[1], ctx_list=ctx, batch_axis=0)
            for x, y in zip(noise, clean):
                # preprocess
                x, y = preprocess(x, y, args["decimation"])
                # forward
                z = net(x)
                for img_noise, img_clean, img_denoise in zip(x, y, z):
                    fig, ax = plt.subplots(1, 3, dpi=100)
                    ax[0].imshow(nd.squeeze(img_noise).asnumpy())
                    ax[1].imshow(nd.squeeze(img_clean).asnumpy())
                    ax[2].imshow(nd.squeeze(img_denoise).asnumpy())
                    ax[0].set_xlabel('noise')
                    ax[1].set_xlabel('clean')
                    ax[2].set_xlabel('denoised')
                    for i in range(3):
                        ax[i].set_xticks([])
                        ax[i].set_yticks([])
                    plot_count += 1
                    plt.savefig(img_folder / f'img{plot_count}.png',
                                bbox_inches='tight')
                    plt.close('all')
        log.ended('Plotting some batches from test set')

    # end top level
    log.ended('Running benchmark em_denoise')
