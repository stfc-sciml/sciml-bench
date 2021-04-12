#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# template.py

# SciML-Bench
# Copyright © 2021 Scientific Machine Learning Research Group
# Scientific Computing Department, Rutherford Appleton Laboratory
# Science and Technology Facilities Council, UK. 
# All rights reserved.

# libs from sciml_bench
from sciml_bench.core.runtime import RuntimeIn, RuntimeOut
from sciml_bench.benchmarks.dms_structure.train import train_model

from pathlib import Path


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
    default_args = {
        # hyperparameters
        'learning_rate': 0.0001,
        'epochs': 30,
        'batch_size': 32,
        'patience': 10,
        # workflow control
        'model_filename': 'model.pt',
        'validation_model_filename': 'valid_best_model.pt',
        'training_history': 'training_history.npy',
    }

    args = smlb_in.bench_args.try_get_dict(default_args)
    smlb_out.activate(rank=0, local_rank=0)
    console = smlb_out.log.console
    dataset = str(list(Path(smlb_in.dataset_dir).glob('**/data-binary.h5'))[0])

    console.begin('Training model')
    train_model(dataset, args, smlb_in, smlb_out)
    console.ended('Training model')
