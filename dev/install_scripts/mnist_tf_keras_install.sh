#!/bin/bash
set -e
set -x

# Create new environment
ENV_NAME=sciml-bench-mnist_tf_keras
conda remove -n $ENV_NAME --all -y --quiet
conda create -n $ENV_NAME python=3.9 -y --quiet
ENV_PATH=$(dirname $(dirname $(which conda)))/envs/$ENV_NAME

# Install conda requirements
conda install -n $ENV_NAME -c conda-forge cudatoolkit=11.2.2 cudnn=8.1.0 -y --quiet
conda install -n $ENV_NAME -c nvidia cuda-nvcc=11.3.58 -y --quiet

# Configure environment variables
mkdir -p $ENV_PATH/etc/conda/activate.d
echo "export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${ENV_PATH}/lib/" >> $ENV_PATH/etc/conda/activate.d/env_vars.sh
echo "export XLA_FLAGS='--xla_gpu_cuda_data_dir=${ENV_PATH}/lib'" >> $ENV_PATH/etc/conda/activate.d/env_vars.sh

# Work around for Ubuntu 22.04. See: https://www.tensorflow.org/install/pip
mkdir -p $ENV_PATH/lib/nvvm/libdevice
cp $ENV_PATH/lib/libdevice.10.bc $ENV_PATH/lib/nvvm/libdevice/

# Install pip requirements
conda run -n $ENV_NAME python -m pip install -q --upgrade pip
conda run -n $ENV_NAME python -m pip install -q "tensorflow==2.11.*" scikit-image
conda run -n $ENV_NAME python -m pip install -q -e . 