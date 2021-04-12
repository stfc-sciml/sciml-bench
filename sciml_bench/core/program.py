#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# program.py

# SciML-Bench
# Copyright © 2021 Scientific Machine Learning Research Group
# Scientific Computing Department, Rutherford Appleton Laboratory
# Science and Technology Facilities Council, UK. 
# All rights reserved.

"""
Program environment of sciml-bench, including
* program config
* register datasets and benchmarks
"""

import yaml
from pathlib import Path


class ProgramEnv:
    """ Class to initialize and store program environment """

    def __init__(self, config_file_path, registration_file_path):
        # --------------
        # program config
        # --------------
        # read config
        with open(config_file_path) as handle:
            cfg = yaml.load(handle, yaml.SafeLoader)
            cfg = {} if cfg is None else cfg

        # dirs
        cfg_dirs = cfg['sciml_bench_dirs']
        self.dataset_root_dir = Path(cfg_dirs['dataset_root_dir']).expanduser()
        self.output_root_dir = Path(cfg_dirs['output_root_dir']).expanduser()

        # --------------------------------
        # register datasets and benchmarks
        # --------------------------------
        # read registration.yml
        with open(registration_file_path) as handle:
            reg = yaml.load(handle, yaml.SafeLoader)
            reg = {} if reg is None else reg

        # datasets
        self.registered_datasets = reg['datasets']

        # benchmarks
        # check dataset registration
        for key, prop in reg['benchmarks'].items():
            assert prop['dataset'] in self.registered_datasets.keys(), \
                f"Dataset of a benchmark is not registered." \
                f"\nBenchmark name: {key}" \
                f"\nDataset name: {prop['dataset']}\n" \
                f'Registered datasets: {list(self.registered_datasets.keys())}'
        self.registered_benchmarks = reg['benchmarks']
