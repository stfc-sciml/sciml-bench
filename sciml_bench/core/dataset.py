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
from sciml_bench.core.config import ProgramEnv
from sciml_bench.core.utils import display_logo

def download(dataset_name: str, dataset_root_dir: Path, prog_env: ProgramEnv, mode):

    if prog_env.is_config_valid() == False:
        print('The configuration file is malformed. Please verify the contents.')
        return

    datasets = prog_env.datasets
    selected_mirror = next(iter(prog_env.mirrors))
    data_mirror = prog_env.mirrors[selected_mirror]

    if dataset_name not in datasets.keys():
        print(f'Dataset {dataset_name} is not part of the SciML-Bench.')
        print(f'Available datasets are: {list(datasets.keys())}')
        return


    # Retransfers are fine  - as we are syncing than copying files 
    
    # create dir
    dataset_dir = (dataset_root_dir / dataset_name).expanduser()
    dataset_dir.mkdir(parents=True, exist_ok=True)


    # Extract the details
    dataset_uri =  's3://' + datasets[dataset_name]['end_point'] + '/' + dataset_name + '/'
    cmd = prog_env.get_download_command(dataset_name)
    cmd = cmd.replace('$SERVER', str(data_mirror))
    cmd = cmd.replace('$DATASET_URI', str(dataset_uri))
    cmd = cmd.replace('$DATASET_DIR', str(dataset_dir))

    if mode == 'background':
        log_dir = Path(prog_env.output_dir / 'download_logs') 
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = str(log_dir) + os.sep + dataset_name + '.log'
        cmd = 'nohup ' + cmd + f' > {log_file} 2>&1 &'
    os.system(cmd)
    return dataset_dir


def is_available(dataset_name, dataset_root_dir: Path):
    dataset_dir = (dataset_root_dir / dataset_name).expanduser()
    return dataset_dir.exists()

def get_status(dataset_name, ENV):
    is_downloaded =  is_available(dataset_name, ENV.dataset_dir) is not False
    str = "Downloaded" if is_downloaded else "Not downloaded"
    return str
