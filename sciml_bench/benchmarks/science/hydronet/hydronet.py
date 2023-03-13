#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# em_denoise.py

# SciML-Bench
# Copyright © 2021 Scientific Machine Learning Research Group
# Scientific Computing Department, Rutherford Appleton Laboratory
# Science and Technology Facilities Council, UK. 
# All rights reserved.


import time
import yaml
import torch
import math
from pathlib import Path
import os, sys
import json, csv
import argparse, shutil
import logging

from torch.utils.tensorboard import SummaryWriter
from torch_geometric.nn import DataParallel
from tqdm import tqdm
import datetime

# Hydronet utils

from sciml_bench.benchmarks.science.hydronet import data

print(">>>> Working directory: ", os.getcwd())
'''
import sciml_bench.benchmarks.science.hydronet.utils.data
import sciml_bench.benchmarks.science.hydronet.utils.models
import sciml_bench.benchmarks.science.hydronet.utils.train
import sciml_bench.benchmarks.science.hydronet.utils.eval
import sciml_bench.benchmarks.science.hydronet.utils.split
import sciml_bench.benchmarks.science.hydronet.utils.hooks
'''

from sciml_bench.core.runtime import RuntimeIn, RuntimeOut




#####################################################################
# Training mode                                                     #
#####################################################################

def sciml_bench_training(params_in: RuntimeIn, params_out: RuntimeOut):
    """
    Entry point for the training routine to be called by SciML-Bench
    """
    default_args = {
        'batch_size': 128,
        'epochs': 2,
        'lr': .01,
        'use_gpu': True
    }    


    # No distributed training in this one
    #params_out.activate(rank=0, local_rank=0)

    # Log top level process
    log = params_out.log.console
    log.begin(f'Running benchmark hydronet on training mode')
    print("Hydronet: Start training")
   

    print("Hydronet: Stop training")
    # End top level
    log.ended(f'Running benchmark hydronet on training mode')
    

#####################################################################
# Inference mode                                                    #
#####################################################################
# model needs to be specified
#
def sciml_bench_inference(params_in: RuntimeIn, params_out: RuntimeOut):
    """
    Entry point for the inference routine to be called by SciML-Bench
    """

    
    log = params_out.log.console
    log.begin(f'Running benchmark hydronet on inference mode')
    print("Hydronet: Start inference")

    
    print("Hydronet: End inference")
    # End top level
    log.ended('Running benchmark em_denoise on inference mode')
