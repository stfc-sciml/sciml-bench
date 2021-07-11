#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# template.py

# SciML-Bench
# Copyright © 2021 Scientific Machine Learning Research Group
# Scientific Computing Department, Rutherford Appleton Laboratory
# Science and Technology Facilities Council, UK. 
# All rights reserved.

"""
A template for benchmark implementation

To build a benchmark named 'exbench' into SciMLBench:

Step 1: Create a main file `sciml_bench/benchmarks/exbench/exbench.py`.
        A benchmark can have as many files as needed, but the main 
        file is the entry point for the framework, which should 
        be the same name as the benchmark name. 

Step 2: Implement `sciml_bench_training()` function in this template.
        This is the entry point for the training process. You may
        skip this step if the exbench is inference-focussed. 

Step 3: Implement `sciml_bench_inference()` function in this template.
        This is the entry point for the inference process. You may
        skip this step if the exbench is training-focussed.

Step 4: Make necessary entries in the config file located at 
        `sciml_bench/etc/configs/config.yml`. 

Step 5: Once fully tested, please contact us for final integration 
        as part of the SciMLBench suite.

You should also consult the documentation around this. 
"""

# libs from sciml_bench
from sciml_bench.core.runtime import RuntimeIn, RuntimeOut


def sciml_bench_training(params_in: RuntimeIn, params_out: RuntimeOut):
    """
    Main entry of `sciml_bench run` for a benchmark instance

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
    print('Hello world! This is a training template.')


def sciml_bench_inference(params_in: RuntimeIn, params_out: RuntimeOut):
    """
    Main entry of `sciml_bench run` for a benchmark instance

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
    print('Hello world! This is an inference template.')