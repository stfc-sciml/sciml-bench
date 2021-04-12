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

To build a benchmark named XYZ into SciML-Benchmarks:
Step 1: implement XYZ in `sciml_bench/benchmarks/XYZ/XYZ.py`,
        using `sciml_bench_run()` in this template as the entry
        point of `sciml-bench run`
Step 2: register XYZ in `sciml_bench/benchmarks/registration.yml`
"""

# libs from sciml_bench
from sciml_bench.core.runtime import RuntimeIn, RuntimeOut


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
    print('Hello world! I am a template.')
