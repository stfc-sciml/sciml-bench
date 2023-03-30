#!/bin/bash
set -e
set -x

# Create new environment
ENV_NAME=sciml-bench-hydronet
conda remove -n $ENV_NAME --all -y --quiet
conda create -n $ENV_NAME python=3.8 -y --quiet
ENV_PATH=$(dirname $(dirname $(which conda)))/envs/$ENV_NAME

# Install conda requirements
conda install -n $ENV_NAME -y --quiet pytorch==1.12.0 cudatoolkit=11.3 -c pytorch -c conda-forge
conda install -n $ENV_NAME -y --quiet -c pyg pyg
conda install -n $ENV_NAME -y --quiet -c conda-forge tensorboard ase fair-research-login h5py tqdm
conda install -n $ENV_NAME -y --quiet -c conda-forge gdown

# Install pip requirements
conda run -n $ENV_NAME python -m pip install -q --upgrade pip
conda run -n $ENV_NAME python -m pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cu113.html
conda run -n $ENV_NAME python -m pip install -q -e .
