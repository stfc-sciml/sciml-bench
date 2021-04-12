#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# dataset.py

# SciML-Bench
# Copyright © 2021 Scientific Machine Learning Research Group
# Scientific Computing Department, Rutherford Appleton Laboratory
# Science and Technology Facilities Council, UK. 
# All rights reserved.

"""
Dataset download and management
"""
from pathlib import Path
import os
from sciml_bench.core.program import ProgramEnv


def download(dataset_name: str, dataset_root_dir: Path, smlb_env: ProgramEnv):
    # check registration
    reg = smlb_env.registered_datasets
    assert dataset_name in reg.keys(), \
        f'Dataset {dataset_name} is not registered.' \
        f'\nRegistered datasets: {list(reg.keys())}'

    # create dir
    dataset_dir = (dataset_root_dir / dataset_name).expanduser()
    dataset_dir.mkdir(parents=True, exist_ok=True)

    download_method = reg[dataset_name]['download_method']
    os.system(download_method.replace('$DATASET_DIR', str(dataset_dir)))
    return dataset_dir


def found(dataset_name, dataset_root_dir: Path):
    dataset_dir = (dataset_root_dir / dataset_name).expanduser()
    return dataset_dir.exists()
