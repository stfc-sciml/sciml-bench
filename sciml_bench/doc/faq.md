# Frequently Asked Questions 


## Installation and Platforms 

1. What are the supported platforms?

The official support is only for the Linux platform. However, the current version of the framework works on MacOS and on WSL2. This may change in future versions. 

1. Does this work on Windows WSL2?


## Benchmarks, Datasets and Models 

1. How do I find more information about a benchmark, or dataset or a model?

You can use the `info` command. For example `sciml-bench info cloud_slstr_ds1` will display relevant info. You do not have to specify the type of entity. The framework will work this out for you. Please use the `sciml-bench info --help` command to see full usage information. 

1. 


## Data Transfers

1. How do we verify whether the files are of right size? 

The framework uses aws behind the scenes to move data. More specifically, we use `aws s3 sync` (which, automatically degrades to `aws s3 cp` as needed), which calculates an MD5 checksum during the sync process with retries. Therefore, we do not, at the present version, verify every object in the dataset against the server-held version. This feature may be implemented in future versions.  

1. Data download is invoked in foreground mode. Can this be done in background mode?

Yes! Staring at the screen during download can be boring. If you are keen on downloading the datasets in background mode,  you can set this using the `--background` option. For example,  `sciml-bench download --background cloud_slstr_ds1`. Please check the download command options using `sciml-bench download --help`.

## Contributions 


## Training and Inference Mode

1. Can I specify nested directories with files for inference

Yes.


## Devices

1. What is the default placement for my benchmark? CPU or GPU ?

This depends on the machine learning framework you use. 

a) On TensorFlow, this is automatic unless it is specified explicitly. By default GPU devices will be given priority, and will be favoured. If no GPU is available, it is run on CPU. If you have more than one GPU in your system, the GPU with the lowest ID will be selected by default. If you would like to run on a specific GPU, the device has to be specified explicitly. 

b) In PyTorch

c) In MXNet


1. What happens when I run my benchmark on a system with multiple GPUs and multiple machines?

If you would like to exploit multiple GPUs, your benchmark has to be explicitly designed for distributed learning/training. There are several ways of doing this, starting with Distributed TensorFlow, PyTorch DistributedDataParallel (DDP), or Horovod. Each of these methods have their own advantages and disadvantages. 

# System-Specific

1. Why NCCL is preferred over other options? 

NCCL backend, by default, performs GPU-to-GPU communication, avoiding the cost of having to go through host memory. This can lead to significant improvement in performance. 


1. I see lot of cluttering outputs when using TensorFlow. How can I stop these.

These cluttering outputs are emitted by the TensorFlow library. This can be stopped by setting the TF_CPP_MIN_LOG_LEVEL flag to 2 as follows:

```sh 
export TF_CPP_MIN_LOG_LEVEL=2
``


https://www.xavor.com/blog/distributed-training-using-horovod-and-keras