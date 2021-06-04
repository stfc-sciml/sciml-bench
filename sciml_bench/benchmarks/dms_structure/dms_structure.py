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

#####################################################################
# Training mode                                                     #
#####################################################################

def sciml_bench_training(params_in: RuntimeIn, params_out: RuntimeOut):
    """
    Main entry for `sciml_bench run` in training mode.

    :param params_in: runtime input of `sciml_bench run`, useful components:
        * params_in.start_time: start time of running as UTC-datetime
        * params_in.dataset_dir: dataset directory
        * params_in.output_dir: output directory
        * params_in.bench_args: benchmark-specific arguments
    :param params_out: runtime output of `sciml_bench run`, useful components:
        * params_out.log.console: multi-level logger on root (rank=0)
        * params_out.log.host: multi-level logger on host (local_rank=0)
        * params_out.log.device: multi-level logger on device (rank=any)
        * params_out.system: a set of system monitors
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

    args = params_in.bench_args.try_get_dict(default_args)
    params_out.activate(rank=0, local_rank=0)
    console = params_out.log.console
    dataset = str(list(Path(params_in.dataset_dir).glob('**/data-binary.h5'))[0])

    console.begin('Training model')
    train_model(dataset, args, params_in, params_out)
    console.ended('Training model')



#####################################################################
# Inference mode                                                    #
#####################################################################

def sciml_bench_inference(params_in: RuntimeIn, params_out: RuntimeOut):
    pass
