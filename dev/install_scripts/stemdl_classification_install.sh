#!/bin/bash
set -e
set -x

# Create new environment
ENV_NAME=sciml-bench-stemdl_classification
conda remove -n $ENV_NAME --all -y --quiet
conda create -n $ENV_NAME python=3.9 -y --quiet
ENV_PATH=$(dirname $(dirname /home/lhs18285/miniconda3/bin/conda))/envs/$ENV_NAME

# Install conda requirements
conda install -n $ENV_NAME -y --quiet pytorch==1.13.1 torchvision==0.14.1 pytorch-cuda=11.6 -c pytorch -c nvidia

# Install pip requirements
conda run -n $ENV_NAME python -m pip install -q --upgrade pip
conda run -n $ENV_NAME python -m pip install -q "pytorch_lightning==1.9.*" scikit-learn tensorboard
conda run -n $ENV_NAME python -m pip install -q .