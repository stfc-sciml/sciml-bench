# Frequently Asked Questions 


## Installation and Platforms 

__*What are the supported platforms?*__

The official support is only for the Linux platform. However, the current version of the framework works on MacOS and on WSL2. This may change in future versions.  As for Windows, it is severely limited by various packages (such as GPUtil etc). 

__*Does the framework/benchmark work on Windows WSL2?*__

Yes.


## Benchmarks, Datasets and Models 

__*How do I find more information about a benchmark, or dataset or a model?*__

You can use the `info` command. For example `sciml-bench info cloud_slstr_ds1` will display relevant info. You do not have to specify the type of entity. The framework will work this out for you. Please use the `sciml-bench info --help` command to see full usage information. 

__*Why should I name my benchmark fully in lower case?*__

To avoid costly mistakes in coding or notations, we recommend all benchmarks to be named (both at the file level and at the implementation level) in lower case. For example, instead of *MNIST_tf_keras*, we recommend using *mnist_tf_keras*.


## Data Transfers

__*How do we verify whether the files are of right size?*__

The framework uses aws behind the scenes to move data. More specifically, we use `aws s3 sync` (which, automatically degrades to `aws s3 cp` as needed), which calculates an MD5 checksum during the sync process with retries. Therefore, we do not, at the present version, verify every object in the dataset against the server-held version. This feature may be implemented in future versions.  

__*Data download is invoked in foreground mode. Can this be done in background mode?*__

Yes! Staring at the screen during download can be boring. If you are keen on downloading the datasets in background mode,  you can set this using the `--background` option. For example,  `sciml-bench download --background cloud_slstr_ds1`. Please check the download command options using `sciml-bench download --help`.

## Contributions 


## Training and Inference Mode

__*Can I specify nested directories with files for inference?*__

Yes.


## Devices

__*What is the default placement for my benchmark? CPU or GPU?*__

This depends on the machine learning framework being used. 

a) TensorFlow: In TensorFlow, GPUs are used if available unless specifically or explicitly controlled by the code. By default GPU devices will be given priority, and will be favoured. If no GPU is available, CPU is opted. If you have more than one GPU in your system, the GPU with the lowest ID will be selected by default. If you would like to run on a specific GPU, the device has to be specified explicitly. 

b) PyTorch:

c) MXNet:


__*What happens when I run my benchmark on a system with multiple GPUs and multiple machines?*__

If you would like to exploit multiple GPUs, your benchmark has to be explicitly designed for distributed learning/training. There are several ways of doing this, starting with Distributed TensorFlow, PyTorch DistributedDataParallel (DDP), or Horovod. Each of these methods have their own advantages and disadvantages. 

## System & Development

__*Why NCCL is preferred over other options?*__

The NVIDIA Collective Communication Library (NCCL) backend, by default, performs GPU-to-GPU communication, avoiding the cost of having to go through host memory. This can lead to significant improvement in performance. 


__*I see a lot of cluttered outputs when using TensorFlow. How can I stop these?*__

These cluttering outputs are emitted by the TensorFlow library and not by the framework. One method is to set the TF_CPP_MIN_LOG_LEVEL variable as follows:

```sh 
export TF_CPP_MIN_LOG_LEVEL=2
```

__*What is Horovod?*__

Horovod is a framework used to support distributed learning in PyTorch, TensorFlow, MXNet, and other frameworks,  and can easily enable scaling our benchmarks across a number of GPUs.  Horovod supports various controllers such as MPI,  and [Gloo](https://bityl.co/79kC) (only one is needed). 

__*Why manual installation of Horovod is recommended?*__

Horovod can be configured in a number of different ways, for instance, to use a specific collective communication libraries, such as [NCCL](https://bityl.co/79k6). The combination of these requirements or constraints are best handled through manual installation, which is outlined [here](https://bityl.co/79kQ).

__*I installed a specific package for a given benchmark manually. Why should I still specify the dependency in the configuration file?*__

Regardless of the installation technique,  relevant dependencies should be specified in the Configuration file. This is to ensure that dependencies are handled by the framework during runtime. 



__*What are the dependencies that should be specified in the Configuration file for Horovod?*__

Assuming benchmark B relies on Horovod, the exact dependency may be around specific bindings you want to use. For example, for TensorFlow and Horovod, it will be `horovod.tensorflow`. Equally, you can used `horovod.torch`,  or `horovod.mxnet` for PyTorch or MXNet bindings. 


__*How can I simplify the usage of benchmarks that rely on distributed learning?*__

Ideally, if you are planning to rely on distributed learning, we  strongly advice you to use the container technology, which provides a means for running our benchmarks on production clusters (where directly running `sciml-bench` or installing various dependencies may not be possible).  


__*When list or run benchmarks, I see the message `The benchmark [...] does not support training. Terminating the execution`. Why is that?*__

Benchmarks can either focus on training or inference or both. One of the reasons for this message can be that the benchmark does not support the requested mode (training or inference). However, in some cases, despite having the relevant implementation, it may still produce this or similar error message. When verifying or running a given benchmark, the framework instantiates the relevant Python code (on training or inference mode).  The training or inference routine may fail for a number of reasons, including, bug in the code, and platform incompatibility of underlying libraries. As a result, the invocation may fail. This is signalled by the call site. Debugging this can be little tricky. If this error is reported on one of our benchmarks, please report this bug.



https://www.xavor.com/blog/distributed-training-using-horovod-and-kerass