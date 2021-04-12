# Installation and Usage: Table of Contents 

- [1. Getting Started](#1-getting-started)
- [2. Setting Up](#2-setting-up)
  * [2.1 System and Software Requirements](#21-system-and-software-requirements)
  * [2.2 Setting up the Framework](#22-setting-up-the-framework)
    + [2.2.1 Default Installation](#221-default-installation)
    + [2.2.2 Custom Installation](#222-custom-installation)
    + [2.2.3 Setting up Container Images](#223-setting-up-container-images)
  * [2.3 Setting up Benchmarks](#23-setting-up-benchmarks)
  * [2.4 Setting up Datasets](#24-setting-up-datasets)
- [3. Using the Suite and Benchmarks](#3-using-the-suite-and-benchmarks)
  * [3.1 Verifying Benchmarks](#31-verifying-benchmarks)
  * [3.2 Running Benchmarks](#32-running-benchmarks)
  * [3.3 Outputs from the Framework](#33-outputs-from-the-framework)

# 1. Getting Started

This part of the manual will guide you on installing and using the benchmark suite.  As stated in the main [README](../README.md), there are three components that underpins benchmarking, namely, 

1. The core framework,
2. benchmarks, and 
3. datasets

These are covered in the following sections. 



# 2. Setting Up



## 2.1 System and Software Requirements

* Supported Operating Systems: Linux (and derivatives).

* Software: Python 3.6-upwards, pip,  and nvidia-smi. 

The framework and the benchmarks are primarily designed for large-scale systems (at least for training), and, as such,  windows or MacOS systems are not officially supported. However, these non-Linux systems can be used in some cases.

The core framework  heavily relies on machine learning frameworks, such as TensorFlow and PyTorch, and as such,  supported Python versions are limited by these frameworks. In general, Python versions 3.0+ are generally supported. We also rely on GPU-specific libraries, such as nvidia-smi for core capabilities. 


## 2.2 Setting up the Framework 

Installing the framework with default settings is very simple. Default installation will rely on preset directories for data, outputs and options for libraries installed during the setup stage. If a custom installation is needed, see Section "Custom Installation" below. 

### 2.2.1 Default Installation 

Although the installation can be performed on the system-wide setting, it is recommended to do that in a virtual or conda environment. 

1. Create a virtual environment and activate that (for example, `conda create --name bench python=3.9` or `python3 -m vent ....` )

2. Download and navigate to the repository folder.

3. Install `sciml-bench` with the following command, accepting all default suggestions:

   ```sh
   pip install .
   ```

4. Once installed, verify the basic functionality of the framework with one or more of the following basic commands:

   ```sh
   # print help messages
   sciml-bench --help
      
   # print About info
   sciml-bench about
   
   # print system information
   sciml-bench sysinfo
   
   # print registered datasets and benchmarks
   sciml-bench list
   ```

### 2.2.2 Custom Installation 

A custom installation may be desirable if you would like to change the data or output directories, or if there are conflicts on dependencies or with system-wide libraries. The data or output directories can easily be changed by modifying the config.yml file `sciml_bench_config.yml` in the `sciml-bench` folder.  An ideal option for handling clashes with system-wide libraries is to use containers, such as singularity, where the installation is sandboxed with minimal impact on performance.   

### 2.2.3 Setting up Container Images 

Another option is to run these benchmarks through containers.  Container technologies, such as [Singularity](https://sylabs.io/)  or [Docker](https://www.docker.com/), permits sandboxed execution of self-contained installation images. This is ideal to run on production- or large-scale systems, and  through submission scripts.  The  source tree includes necessary configuration files for building both the Docker and Singularity images. These can be found inside the  `config` folder of the source tree. 

**Building a Docker Image** 

The configuration file for building the Docker image, the `Dockerfile`, does not support multi-node execution. In other words, the resulting Docker image, in its present form, can only be used on a single host (but multiGPU support is enabled).  The docker image can be built by 

```sh 
sudo docker build -t sciml-bench -f config/Dockerfile .
```

This will produce the  Docker image, `sciml_bench`. Please see Section 3.3 for using Docker images for running the benchmarks. 

**Building a Singularity Image**

The Singularity container is built in two stages: In the first,  a container with *openmpi4* compatible with the host node is created.  This is then used in the second stage to build a  container with all other `sciml-bench`-specific  dependencies. The Singularity image(s) can be built by running the following set of commands. 

```sh 
singularity build ompi4.sif config/ompi4.def
singularity build sciml-bench.sif config/sciml-bench.def
```

Please check and modify the `def` files any platform-specific aspects as necessary. This build process may take a while, producing a container named  `sciml-bench.sif` which can be used to run the benchmarks. Please see Section 3.3 for using Singularity images for running the benchmarks. 

## 2.3 Setting up Benchmarks

Benchmarks supplied by the suite can either be installed at once, or selectively. 

* To install  all  benchmarks:

  ```sh
  sciml-bench install all
  ```

* To install a selected benchmark, for example, to install the em_denoise benchmark

  ```sh
  sciml-bench install em_denoise
  ```

* To install more than one benchmark, separate them using commas, e.g, 

  ```sh
  sciml-bench install em_denoise,dms_scatter
  ```

Each benchmark in the suite has its own dependencies, and these dependencies are automatically installed. We make our best efforts in ensuring that these dependencies do not conflict each other, but there can be cases of conflicts. In such cases, you may be required to install these manually (and may have to follow the on-screen instructions). 


## 2.4 Setting up Datasets

Each benchmark relies on one or more datasets, and in particular on specific version of the datasets, if applicable. These datasets are  not supplied along with the suite by default. Instead, they must be downloaded explicitly.  

* Similar to benchmarks, datasets can also be downloaded collectively, or selectively  

  ```sh
  sciml-bench download all
  ```

  or

  ```sh
  sciml-bench download MNIST
  ```

  or

  ```sh
  sciml-bench download dms_sim, em_noise_sim, mnist
  ```

* Relevant datasets can be queried through the `list` command as follows: 

  ```sh
  sciml-bench list datasets
  ```

The dataset names are case sensitive, and are, by default, downloaded to the default directory.  The default directory can be overridden through two different methods: (1) by modifying the `sciml_bench_config.yml` file, or (2) by using an optional argument  `--dataset_root_dir` when  using the download command.  Additional options for the download command can be found via the `sciml-bench download --help` command.

# 3. Using the Suite and Benchmarks

## 3.1 Verifying Benchmarks

Before running a benchmark,  any issues with the relevant datasets and modules can be verified via the verify flag. 

```sh
sciml-bench list --verify
```

The results from the verify flag indicates whether the relevant dataset(s) has/have been downloaded,  and whether the dependencies of the benchmark are fully satisfied. If the verify command outputs  `Modules verified: False` for any benchmark, the relevant datasets or dependencies have to be downloaded /  installed.  


## 3.2 Running Benchmarks

**Running Benchmarks from the Source Tree**

The framework provides the `sciml-bench run` command to  run a single benchmark. This command can directly be used to run the benchmark interactively. However, this may not be possible always (such as on production systems). If this is the case, please consult the section that follows this.

If the benchmarks were to be run  interactively,  the`sciml-bench run` command  can be used. Please execute the command  `sciml-bench run --help`  to see the full list of options. 


```
Usage: sciml-bench run [OPTIONS] BENCHMARK_NAME

			 Runs a single benchmark.

Possible options:
  --dataset_dir TEXT              Directory of dataset.
                                  Default: dataset_root_dir/dataset_name/
                                           (dataset_root_dir in sciml_bench_config.yml).

  --output_dir TEXT               Output directory of this run.
                                  Convention: use --output_dir=@foo to save outputs under
                                              output_root_dir/benchmark_name/foo/;
                                              without "@", foo is used as a normal path.  

  --monitor_on / --monitor_off    Monitor system usage during runtime.
                                  Default: True.

  --monitor_interval FLOAT        Time interval for system monitoring.
                                  Default: 1.0.

  --monitor_report_style [pretty|yaml|hdf5]
                                  Report style of system monitor.
                                  Default: pretty.

  -b, --benchmark_specific <TEXT TEXT>...
                                  Benchmark-specific arguments.
                                  Usage: -b key1 val1 -b key2 val2 ...

  --help                          Show this message and exit.
```

The options are elaborated below:

* `--dataset_dir`: the dataset directory, useful when the dataset is 
  stored somewhere other than the default location.
* `--output_dir`: the output directory, either absolute or relative to 
  the current directory, required.
  By convention, you can use `--output_dir=@foo` to set the directory to
  `output_root_dir/benchmark_name/foo/` (it will be `./foo/` without the `@` prefix).
* `--monitor_*`: system monitor options. 
  During the runtime, `sciml-bench` will automatically
  monitor system usage (such as CPU, memory, disk and GPU usage) and 
  generate reports at the end of the run. We strive to minimize the overheads for system monitoring. If such overheads turn out noticeable, you can switch off the system monitor using the
  `--monitor_off` option or increase the time interval between snapshots using the 
  `--monitor_interval` option, which is, by default, set to 1 second interval.

* `--benchmark-specific` or `-b`: benchmark-specific arguments,
  Can be used to specify benchmark-specific parameters, such as hyper-parameters, system configurations and workflow control.  Each argument to the `-b` command must be specified in the form key value pair. For example,  `-b epochs 50` and `-b batch_size 16`. 



**Running Benchmarks Using Container Images**

If the benchmarks were to be run on a production cluster, ideally they may have to be run using one of the container technologies, such as Singularity of Docker.  Section 2.2.3 outlined how these images can be built Assuming that relevant images are in place,  

**Using the Docker Image** 

Assuming that the Docker image is named `sciml-bench`, the standard commands of the benchmark framework can be run by simply prefixing them with `sudo docker run` command. For example, the following are some examples: 

```sh 
sudo docker run --gpus all -v /tmp/bench_data:/root/bench_data sciml-bench list

sudo docker run --gpus all -v /tmp/bench_data:/root/bench_data sciml-bench download MNIST

sudo docker run --gpus all -v /tmp/bench_data:/root/bench_data sciml-bench run MNIST_torch --output_dir=test-mnist
```

Here, the `-v` option here helps mounting the datasets into the host operating system. 

**Using the Singularity Image**

Assuming that the Singularity image is named `sciml-bench.sif`, the standard commands of the benchmark framework can be run by simply prefixing them with `singularity run` command. For example, the following are some examples: 

```sh
singularity run --nv sciml-bench.sif list
singularity run --nv sciml-bench.sif download MNIST
singularity run --nv sciml-bench.sif run MNIST_torch \
                --output_dir=test-mnist
```

Here, relevant folders are automatically accessible inside the host operating system. 

**Using with Third-Party Applications**

In addition to running the framework directly or through containers, it can also be passed on to third-party applications, such as `horovodrun`, which is essential when run across multiple nodes or even multiple GPUs. For example, 

```sh
horovodrun -np 2 sciml-bench run MNIST_torch --output_dir=demo_output__MNIST_torch --monitor_interval=0.1 -b use_cuda false -b epochs 2
```

##  3.3 Outputs from the Framework 

Outputs of the benchmark runs are automatically logged into a sub-directory inside the main output directory, which is specified in the `sciml_bench_config.yml` file. This is, by default, set to `~/sciml_bench_user/outputs/` but can be overridden during installation.  The sub-directory name is same as the benchmark name. All the arguments used for a run are saved to the `arguments_used.yml` file in this  directory. The sub-directory will also contain all log files capturing the outputs of the run. 

<div style="text-align: right"> ◼︎ </div>


