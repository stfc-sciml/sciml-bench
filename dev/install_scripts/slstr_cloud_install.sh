#!/bin/bash
set -e
set -x

# Create new environment
ENV_NAME=sciml-bench-slstr_cloud
conda remove -n $ENV_NAME --all -y --quiet
conda create -n $ENV_NAME python=3.9 -y --quiet
ENV_PATH=$(dirname $(dirname $(which conda)))/envs/$ENV_NAME

# Install conda requirements
conda install -n $ENV_NAME --quiet -y -c conda-forge cudatoolkit=11.2.2 cudnn=8.1.0
conda install -n $ENV_NAME --quiet -y -c nvidia cuda-nvcc=11.3.58
conda install -n $ENV_NAME --quiet -y -c conda-forge -c nvidia cudatoolkit-dev nccl
conda install -n $ENV_NAME --quiet -y -c conda-forge mpi4py cmake

# Configure environment variables
mkdir -p $ENV_PATH/etc/conda/activate.d
echo "export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${ENV_PATH}/lib/" >> $ENV_PATH/etc/conda/activate.d/env_vars.sh
echo "export XLA_FLAGS='--xla_gpu_cuda_data_dir=${ENV_PATH}/lib'" >> $ENV_PATH/etc/conda/activate.d/env_vars.sh

# Work around for Ubuntu 22.04. See: https://www.tensorflow.org/install/pip
mkdir -p $ENV_PATH/lib/nvvm/libdevice
cp $ENV_PATH/lib/libdevice.10.bc $ENV_PATH/lib/nvvm/libdevice/

# Install pip requirements
conda run -n $ENV_NAME python -m pip install -q --upgrade pip
conda run -n $ENV_NAME python -m pip install -q "tensorflow==2.11.*" scikit-learn h5py
conda run -n $ENV_NAME python -m pip install -q "horovod[tensorflow]"
conda run -n $ENV_NAME python -m pip install -q -e . 
