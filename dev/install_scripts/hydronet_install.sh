#!/bin/bash
set -e
set -x

# Create new environment
ENV_NAME=sciml-bench-hydronet
# Directory of this script
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

conda remove -n $ENV_NAME --all -y --quiet
conda create -n $ENV_NAME  -y --quiet --file $SCRIPT_DIR/hydronet_deps.txt
