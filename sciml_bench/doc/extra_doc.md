# Installing Horovod 

[Horovod](https://horovod.ai/) is a framework to support distributed learning in PyTorch, TensorFlow, MXNet, and other frameworks,  and can easily enable scaling our benchmarks across a number of GPUs.  Horovod supports various collective communication libraries, such as MPI,  [NCCL](https://bityl.co/79k6) and [Gloo](https://bityl.co/79kC) (only one is needed). Ideally, if you are planning to rely on distributed learning, we  strongly advice you to use container technology, which provides a means for running our benchmarks on production clusters (where directly running `sciml-bench` or installing various dependencies may not be possible). 

If you are not using containers, you can either install Horovod manually (which is recommended) or through the `sciml-bench install` command. If horovod installation is attempted through SciMLBench, Horovod dependencies (such as `horovod.torch`, `horovod.tensorflow` or `horovod.mxnet`) are extracted from the Configuration file. Regardless whether the Horovod is installed manually or through SciMLBench, these dependencies are expected to be specified in the Configuration file. Please consult the [Configuration Options] section for more details. 

## Manual Installation of Horovod 

This is the **recommended** way to install Horovod. To install Horovod manually, please consult the Horovod [Installation Page](https://bityl.co/79kQ). Any dependencies on Horovod must be declared for each benchmark. 


## Installation of Horovod through SciMLBench 

If the installation is attempted through SciMLBenchIn essence, SciMLBench will automatically extract any Horovod dependencies from the configuration file and will attempt to install necessary bindings (for TensorFlow or PyTorch or  MXNet). Although the framework makes the best effort, horovod dependencies are usually system-specific and may require re-installation attempts to satisfy, which may leave the sandbox (or the environment) in an unusable state if it fails. However, the framework assumes that user is trying to do this inside an environment, and hence the damage is minimal.  

# Configuration Options 

SciMLBench relies on a single configuration file, `config.yml`, inside the `$SCIMLBENCHROOT/etc/configs/` folder. The configuration file has the following sections:

1. Data Mirrors, 
2. Download Commands, 
3. Directories, 
4. Datasets, and 
5. Benchmarks.

Each of these sections contains subsections or (key:value) pairs.  These are discussed in the following subsections.

## Data Mirrors

SciMLBench datasets are mirrored at various locations, and this section includes the URLs for those mirrors. Usually, these are not supposed to be modified. However, if you are developing a benchmark, you can add your own servers. The framework decides which mirrors based on the bandwidth information, but this can be overridden. 

## Download Commands

A number of different types of download commands can be defined, and then can be used in latter sections / subsections. Some examples of download commands are: 

```yaml
    download_command1: "aws s3 --no-sign-request --endpoint-url $SERVER sync $DATASET_URI $DATASET_DIR"
    download_command2: "wget -c https://www.dropbox.com/s/6xhviply27tw3yu/mnist.tar.bz2 -O -
            | tar -jx -C $DATASET_DIR"
```

where variables like `$SERVER`, `$DATASET_URI` and `$DATASET_DIR` refers to one of the data mirrors (defined above), location of the dataset in the mirror, and local directory where the dataset should be saved (see below). 


## Directories 

This section specifies where the downloaded datasets, outputs of the benchmark runs and downloaded pre-trained models, to be saved locally on the system, respectively.  An example entry would look like the following:

```yaml
directories:
    dataset_root_dir: ~/sciml_bench/datasets
    output_root_dir: ~/sciml_bench/outputs
    models_dir: ~/sciml_bench/models
```
In the case of containers, these directories can be specified as part of the run. 

## Datasets 


## Benchmarks 

Benchmarks are included in the main section `benchmarks`. Each benchmark can have following sub-sections:

* **datasets**: These are dataset(s) used by the current benchmark. A benchmark can rely on one or more datasets, and can be specified using comma separated values. 
* **is_example**: Possible Values are `True` or `False`. This flag indicates whether this benchmark is an example / demo benchmark. Functionally, this does not make any  difference, but can be listed separately using the `list` command and can be used as templates for building complex benchmarks. 
* **dependencies**: These are actual library dependencies that the current benchmark relies on specified inside a string in a comma separated form.
* **types**: Specifies the type of benchmark. 

Each of these aspects are discussed in detail below. 

### Datasets 


### Dependencies 

 An example entries are: `torch, tensorflow, scikit-learn,horovod.torch`. Each of these dependencies must be installable using the `pip install` command. Horovod-based dependencies must be paid a special attention. Please see [Installing Horovod] section for more details on this.  These dependencies can also be platform specific. For instance, if using MacOS for inference, installing `mxnet-cuXXX` is likely to fail. In this case, specifying the `mxnet` as the dependency is likely to succeed. 


### Supported Benchmark Types

Benchmarks can focus on training, inference or both. Although most of the benchmarks covers both the training and inference, some benchmarks can be focussed on only one of the aspects. 
This is specified using the `type` section within each benchmark using a comma separated values of a string. An example section of the Configuration file is:

```yaml
benchmarks:
    dms_structure:
        datasets: dms_sim
        dependencies: 'torch'
        type: 'training, inference'
```

Here, the `dms_structure` benchmark covers both the training types. The `type` string can be any of *'training, inference'*, *'inference'*,  or *'training'*. If none specified, both training and inference are assumed. The order does not matter. 


# Running Benchmarks 

Benchmarks can be run either in training mode or in inference mode. 



