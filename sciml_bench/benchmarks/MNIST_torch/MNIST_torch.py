#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# MNIST_torch.py

# SciML-Bench
# Copyright © 2021 Scientific Machine Learning Research Group
# Scientific Computing Department, Rutherford Appleton Laboratory
# Science and Technology Facilities Council, UK. 
# All rights reserved.

"""
Benchmark: MNIST_torch
Classifying MNIST using a CNN implemented with pytorch.

NOTES:
* This is an example of how to build a benchmark into SciML-Benchmarks.
* It is registered in registration.yml as MNIST_pytorch.
* In this example, we demonstrate how to
  1) handle complex benchmark-specific arguments such as workflow control;
  2) activate and use smlb_out for distributed learning;
  3) implement a benchmark with multiple files.
* This example is adapted from
  github.com/horovod/horovod/blob/master/examples/pytorch/pytorch_mnist.py
"""

# libs from sciml_bench
from sciml_bench.core.runtime import RuntimeIn, RuntimeOut

# libs required by implementation
from pathlib import Path
import yaml
import torch
import horovod.torch as hvd

# implementation from other files
from sciml_bench.benchmarks.MNIST_torch.impl_mnist_torch \
    import create_dataset_sampler_loader, MNISTNet, train, predict


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

    # -------------------------------
    # initialize horovod and smlb_out
    # -------------------------------
    # initialize horovod
    hvd.init()

    # initialize smlb_out with hvd.rank() and hvd.local_rank()
    # Note: in this example, we will log the concurrent sub-processes on
    #       console and the parallelized training loop on devices,
    #       leaving the host logger unactivated.
    smlb_out.activate(rank=hvd.rank(), local_rank=hvd.local_rank(),
                      activate_log_on_host=False,
                      activate_log_on_device=True, console_on_screen=True)

    # top-level process
    console = smlb_out.log.console
    console.begin('Running benchmark MNIST_torch')
    console.message(f'hvd.rank()={hvd.rank()}, hvd.size()={hvd.size()}')

    # -------------------------------------
    # input arguments and torch environment
    # -------------------------------------
    console.begin('Handling input arguments and torch environment')
    # default arguments
    default_args = {
        # torch env
        'seed': 1234, 'use_cuda': True,
        # network parameters
        'n_filters': 32, 'n_unit_hidden': 128,
        # hyperparameters
        'epochs': 2, 'batch_size': 64, 'loss_func': 'CrossEntropyLoss',
        'optimizer_name': 'Adam', 'lr': 0.001,
        'batch_size_test': 2000,  # use a large one for validation
        'batch_log_interval': 100,  # interval to log batch loop
        # workflow control
        'load_weights_file': '',  # do training if this is empty
    }
    # replace default_args with bench_args
    # Note: try_get_dict() provides a way to do try_get() collectively.
    args = smlb_in.bench_args.try_get_dict(default_args=default_args)

    # torch environment
    torch.manual_seed(args['seed'])
    console.message(f'Random seed: {args["seed"]}')
    if args['use_cuda']:
        if torch.cuda.is_available():
            # Horovod: pin GPU to local rank
            torch.cuda.set_device(hvd.local_rank())
            console.message('CUDA: available.')
            # Horovod: limit # of CPU threads to be used per worker
            torch.set_num_threads(1)
        else:
            args['use_cuda'] = False
            console.message('WARNING: CUDA is unavailable, '
                            'setting argument "use_cuda" to False')

    # save actually used arguments
    if hvd.rank() == 0:
        args_file = smlb_in.output_dir / 'arguments_used.yml'
        with open(args_file, 'w') as handle:
            yaml.dump(args, handle)
        console.message(f'Arguments used are saved to:\n{args_file}')
    console.ended('Handling input arguments and torch environment')

    # --------------------------------
    # data sampler, data loader, model
    # --------------------------------
    # data
    console.begin('Creating data sampler and loader')
    dataset_dir = smlb_in.dataset_dir
    train_set, train_sampler, train_loader = create_dataset_sampler_loader(
        dataset_dir / 'train.hdf5',
        args['use_cuda'], args['batch_size'], hvd)
    test_set, test_sampler, test_loader = create_dataset_sampler_loader(
        dataset_dir / 'test.hdf5',
        args['use_cuda'], args['batch_size_test'], hvd)
    console.message(f'Dataset directory: {dataset_dir}')
    console.message(f'Batch size: {args["batch_size"]}')
    console.ended('Creating data sampler and loader')

    # model
    with console.subproc('Creating CNN model'):
        model = MNISTNet(args['n_filters'], args['n_unit_hidden'])
        console.message(f'# convolutional filters: {args["n_filters"]}')
        console.message(f'# units in hidden layer: {args["n_unit_hidden"]}')

    # -------------------------------------
    # Train model or load weights from file
    # -------------------------------------
    if args['load_weights_file'] == '':
        console.begin('Training model')
        # stamp train() in system monitor
        smlb_out.system.stamp_event('start train()')
        # train model
        history = train(model,
                        train_sampler, train_loader, test_sampler, test_loader,
                        use_cuda=args['use_cuda'], epochs=args['epochs'],
                        loss_func=args['loss_func'],
                        optimizer_name=args['optimizer_name'], lr=args['lr'],
                        batch_log_interval=args['batch_log_interval'],
                        hvd=hvd, smlb_out=smlb_out)
        # save weights and history
        if hvd.rank() == 0:
            # weights
            weights_file = smlb_in.output_dir / 'model_weights.h5'
            torch.save(model.state_dict(), weights_file)
            console.message(f'Model weights saved to:\n{weights_file}')
            # history
            history_file = smlb_in.output_dir / 'training_history.yml'
            with open(history_file, 'w') as handle:
                yaml.dump(history, handle)
            console.message(f'Training history saved to:\n{history_file}')
        console.ended('Training model')
    else:
        # load weights from file
        with console.subproc('Loading model weights (Training skipped)'):
            weights_file = Path(args['load_weights_file']).expanduser()
            model.load_state_dict(torch.load(weights_file))
            console.message(f'Weights loaded from:\n{weights_file}')
        # the device log will be empty, so better say something
        smlb_out.log.device.message('Training skipped, no message here.')

    # predict
    if hvd.rank() == 0:
        # stamp predict() in system monitor
        smlb_out.system.stamp_event('start predict()')
        with console.subproc('Making predictions on test set (on root only)'):
            pred = predict(model, test_set.images, args['use_cuda'],
                           to_classes=True)
            correct = pred.eq(test_set.labels).sum()
            console.message(f'{correct} correct predictions '
                            f'for {len(pred)} images '
                            f'(accuracy: {correct / len(pred) * 100:.2f}%)')

    # top-level process
    console.ended('Running benchmark MNIST_torch')
