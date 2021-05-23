FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

ENV PYTHON_VERSION=3.8
ENV TENSORFLOW_VERSION=2.3.0
ENV PYTORCH_VERSION=1.7.1+cu101

RUN apt-get -y update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
        build-essential \
        cmake \
        git \
        curl \
        vim \
        wget \
        libcudnn8 \
        libnccl2 \
        libnccl-dev \
        ca-certificates \
        libjpeg-dev \
        libpng-dev \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        python${PYTHON_VERSION}-distutils

RUN ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py


# Install TensorFlow, Keras and PyTorch
RUN pip install torch==${PYTORCH_VERSION} torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install mxnet-cu101 tensorflow-gpu==${TENSORFLOW_VERSION} keras h5py filelock matplotlib

# Install Horovod, temporarily using CUDA stubs
RUN ldconfig /usr/local/cuda-10.1/targets/x86_64-linux/lib/stubs && \
    HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_WITH_MXNET=1 HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 pip install --no-cache-dir horovod && \
    ldconfig


RUN echo NCCL_DEBUG=INFO >> /etc/nccl.conf && \
    echo NCCL_SOCKET_IFNAME=^docker0 >> /etc/nccl.conf


# Copy over sciml-bench files
COPY sciml_bench /sciml-benchmarks/sciml_bench
COPY requirements.txt /sciml-benchmarks/requirements.txt
COPY MANIFEST.in /sciml-benchmarks/MANIFEST.in
COPY setup.py /sciml-benchmarks/setup.py
COPY doc /sciml-benchmarks/doc

RUN cd /sciml-benchmarks && pip install .

ENTRYPOINT [ "sciml-bench" ]
