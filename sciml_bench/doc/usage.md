# Installation and Usage: Table of Contents 

# 1. Getting Started

This part of the manual will guide you on installing and using the benchmark suite.  As stated in the main [README](../README.md).  There are three components that underpins benchmarking, namely, 

1. The core framework,
2. benchmarks, and 
3. datasets

These are covered in the following sections. 



# 2. Setting Up


## 2.1 System and Software Requirements

* Supported Operating Systems: Linux (and derivatives).

* Software: Python 3.8-upwards, pip, and nvidia-smi

* Benchmark-specific Software: Please consult the benchmarks.


The framework and the benchmarks are primarily designed for large-scale systems (at least for training), and, as such,  windows or MacOS systems are not officially supported. However, these non-Linux systems can be used in some cases, for instance in inference mode.

Although the core framework  does not heavily rely on machine learning frameworks, such as TensorFlow, PyTorch, or MXNet, benchmarks may rely on one or more of these. As such,  supported Python versions are limited by these frameworks. In general, Python versions 3.8+ are generally supported. We also rely on GPU-specific libraries, such as nvidia-smi for core capabilities. 

Benchmarks may rely on distributed learning, and subsequently, dependencies on distributed learning frameworks, such as TensorFlow Distributed, PyTorch Distributed Data Paralle (DDP) or Horovod, are inevitable. Although some of these can be installed through our framework, these frameworks may have complex dependencies and in some cases, OS-level installations. As such, we recommend manual installation of these frameworks or we recommend using containers. 


## 2.2 Setting up the Framework (Default Installation)

Installing the framework with default settings is very simple. Default installation  relies on preset directories for data, outputs and options for libraries installed during the setup stage. If a custom installation is needed, see Section "Custom Installation" below. 

Although the installation can be performed on the system-wide setting, it is recommended to do that in a virtual or conda environment (also known as the sandboxed environment). This way, any damages are minimal and contained.

1. Create a virtual environment and activate that (for example, `conda create --name bench python=3.9` or `python3 -m vent ....` )

2. Download and navigate to the repository folder.

3. Install `sciml-bench` with the following command, accepting all default suggestions:

   ```sh
   pip install  --use-feature=in-tree-build .
   ```

4. Once installed, verify the basic functionality of the framework with one or more of the following basic commands:

   ```sh
   # print the verson 
   sciml-bench --version
      
   # print help messages
   sciml-bench --help

   # print About info
   sciml-bench about
   
   # print system information
   sciml-bench sysinfo
   
   # print registered datasets and benchmarks
   sciml-bench list
   ```

### 2.3 Setting up the Framework (Custom Installation) 

A custom installation may be desirable if you would like to change the data or output directories, or if there are conflicts on dependencies or with system-wide libraries. The data or output directories can easily be changed by modifying the Configuration file `etc/configs/config.yml`. 


### 2.4 Installing Horovod 

[Horovod](https://horovod.ai/) is a framework to support distributed learning in PyTorch, TensorFlow, MXNet, and other frameworks,  and can easily enable scaling our benchmarks across a number of GPUs.  Horovod supports various controllers
such as MPI,  and [Gloo](https://bityl.co/79kC) (only one is needed). Ideally, if you are planning to rely on distributed learning, we  strongly advice you to use container technology, which provides a means for running our benchmarks on production clusters (where directly running `sciml-bench` or installing various dependencies may not be possible). Furthrmore, horovod can be configured to use collective communication libraries, such as [NCCL](https://bityl.co/79k6). The combination of these are best handled through manual installation. 

If you are not using containers, we strong suggest you to  install Horovod manually. Relevant Horovod dependencies (such as `horovod.torch`, `horovod.tensorflow` or `horovod.mxnet`) are specified in the Configuration file. This is to ensure that dependencies are handled well. You can also try installing Horovod through `sciml-bench install b1` command, where Horovod is a dependency for the benchmark `b1`. Although this is, in theory, possible,  the complex dependency patterns of Horovod are not handled inside the SciML-Bench.  Please consult the [Configuration Options]() section for more details. 

Manual installation  is the **recommended** way to install Horovod. To install Horovod manually, please consult the Horovod [Installation Page](https://bityl.co/79kQ). If the installation is attempted through SciMLBench, SciMLBench will attempt to install necessary bindings (for TensorFlow or PyTorch or  MXNet) for Horovod. Although the framework makes the best effort, Horovod dependencies are usually system-specific and may require re-installation attempts to satisfy, which may leave the sandbox (or the environment) in an unusable state if it fails. When done so, the damage is minimal.  

An ideal option for handling clashes with system-wide libraries is,  using containers, such as Singularity, where the installation is sandboxed with minimal impact on performance.   

### 2.5 Setting up Container Images 

When using the benchmark suite on a production cluster, we recommend using containers.  Container technologies, such as [Singularity](https://sylabs.io/)  or [Docker](https://www.docker.com/), permits sandboxed execution of self-contained installation images. This is ideal to run on production- or large-scale systems, and  through submission scripts.  The  source tree includes necessary configuration files for building both the Docker and Singularity images. These can be found inside the  `/etc/recipes` folder. 

**Building a Docker Image** 

The configuration file for building the Docker image, the `Dockerfile`, does not support multi-node execution. In other words, the resulting Docker image, in its present form, can only be used on a single host (but multiGPU support is enabled).  The docker image can be built by 

```sh 
sudo docker build -t sciml-bench -f etc/recipes/Dockerfile .
```

This will produce the  Docker image, `sciml_bench`. Please see Section 3.3 for using Docker images for running the benchmarks. 

**Building a Singularity Image**

The Singularity container is built in two stages: In the first stage,  a container with *openmpi4* compatible with the host node is created.  This is then used in the second stage to build a  container with all other `sciml-bench`-specific  dependencies. The Singularity image(s) can be built by running the following set of commands. 

```sh 
singularity build ompi4.sif etc/recipes/ompi4.def
singularity build sciml-bench.sif etc/recipes/sciml-bench.def
```

Please check and modify the `def` files any platform-specific aspects as necessary. This build process may take a while, producing a container named  `sciml-bench.sif` which can be used to run the benchmarks. Please see Section 3.3 for using Singularity images for running the benchmarks. 

## 2.6 Setting up Benchmarks

Benchmarks supplied by the suite can either be installed at once, or selectively. 

* To install all benchmarks:

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


## 2.7 Setting up Datasets

Each benchmark relies on one or more datasets, and in particular on specific version of the datasets, if applicable. As most of these datasets are considerably large in size, they are not supplied along with the suite by default. Instead, they must be downloaded explicitly.  

* Similar to benchmarks, datasets can also be downloaded collectively, or selectively  

  ```sh
  sciml-bench download all
  ```

  or

  ```sh
  sciml-bench download mnist
  ```

  or

  ```sh
  sciml-bench download dms_sim, em_noise_sim, mnist
  ```

* Relevant datasets can be queried through the `list` command as follows: 

  ```sh
  sciml-bench list datasets
  ```

The dataset names are case sensitive, and are, by default, downloaded to the default directory.  The default directory can be overridden through two different methods: (1) by modifying the `etc/configs/config.yml` file, or (2) by using an optional argument  `--dataset_root_dir` when  using the download command.  Additional options for the download command can be found via the `sciml-bench download --help` command.

Furthermore, by default, the `download` command operates in foreground mode. However, this can be set to background mode if desired, particularly, when downloading large datasets. This can be achieved as follows:

```sh
sciml-bench download --mode background all
```

Finally, the datasets can be verified to see whether they have been downloaded or not as follows:

 ```sh
  sciml-bench list datasets --verify
 ```

This, however, does not check the completeness or the integrity of the downloaded files.

# 3. Using the Suite and Benchmarks

## 3.1 Verifying Benchmarks

Before running a benchmark,  any issues with the relevant datasets and modules can be verified via the verify flag. 

```sh
sciml-bench list --verify
```

The results from the verify flag indicates whether the relevant dataset(s) has/have been downloaded,  and whether the dependencies of the benchmark are fully satisfied. If the verify command outputs  *Not Runnable* for any benchmark, the relevant datasets dependencies have to be downloaded /  installed. The *Not Runnable*  output can also be the result of benchmark codes not being able to import the depenendencies.


## 3.2 Running Benchmarks

**Running Benchmarks from the Source Tree**

The framework provides the `sciml-bench run` command to  run a single benchmark. This command can directly be used to run the benchmark interactively. However, this may not be possible always (such as on production systems or where job submission is necessary). If this is the case, please use containers. 

If the benchmarks were to be run  interactively,  the`sciml-bench run` command  can be used. Please execute the command  `sciml-bench run --help`  to see the full list of options. 


```
Usage: sciml-bench run [OPTIONS] BENCHMARK_NAME

  Run a given benchmark on a training/inference mode.

Options:
  --mode [training|inference]     Sets the mode to training or
                                  inference.
                                  Default: training.
  --model TEXT                    Sets the model(s) to be used 
                                  (only for inference.)
                                  If not specified, framework will attempt to find the model(s) in the models directory
                                  Default: None.    
  --dataset_dir TEXT              Directory for the dataset(s).
                                  Default: dataset directory from the config file
  --output_dir TEXT               Output directory for this run.
                                  If not specified, outputs will be logged under
                                          output_root_dir/benchmark_name/yyyymmdd/
                                  where a yyyymmdd represents the current date
                                  Use --output_dir=@foo to save outputs under
                                          output_root_dir/benchmark_name/foo/
                                  If "@" is omitted, absolute path is assumed.
  --monitor_on / --monitor_off    Monitor system usage during 
                                  runtime.
                                  Default: Monitor On.
  --monitor_interval FLOAT        Time interval for system
                                  monitoring.
                                  Default: 1.0s.
  --monitor_report_style [pretty|yaml|hdf5]
                                  Report style for system monitor.
                                  Default: Pretty.
  -b, --benchmark_specific <TEXT TEXT>...
                                  Benchmark-specific arguments.
                                  Usage: -b key1 val1 -b key2 val2 ...
  --help                          Show this message and exit.

```

The options are elaborated below:

* `--mode`: specifies whether the benchmark is to be run on training mode or inference mode. 
* `--model`: This option applies only if the inference mode is specified, and this an avenue for specifying the model file for 
performing the inference option. It is also possible to specify multiple model files by repeating the option in the form of `--model file1 --model file2`
* `--dataset_dir`: the dataset directory, useful when the dataset is stored somewhere other than the default location.
* `--output_dir`: the output directory, either absolute or relative to the current directory, required. By convention, you can use `--output_dir=@foo` to set the directory to
  `output_root_dir/benchmark_name/foo/` (it will be `./foo/` without the `@` prefix). If not specified, the framework will create a folder with the current date.
* `--monitor_*`: system monitor options. 
  During the runtime, `sciml-bench` will automatically
  monitor system usage (such as CPU, memory, disk and GPU usage) and generate reports at the end of the run. We strive to minimize the overheads for system monitoring. If such overheads turn out noticeable, you can switch off the system monitor using the
  `--monitor_off` option or increase the time interval between snapshots using the 
  `--monitor_interval` option, which is, by default, set to 1 second interval.

* `--benchmark-specific` or `-b`: benchmark-specific arguments,
  Can be used to specify benchmark-specific parameters, such as hyper-parameters, files, system configurations and other workflow controls.  Multiple arguments can be specified by repeating the `-b` option, in the form of `-b epochs 50 -b batch_size 16`. 

Please see the benchmarks for example usage of some of these options, such as `--model` or `-b`. 


**Running Benchmarks Using Container Images**

If the benchmarks were to be run on a production cluster, ideally they may have to be run using one of the container technologies, such as Singularity of Docker. Although both methods need elevated previleges (equivalent to root access) for building containers, Singularity does not demand elevated previleges for running the containers. Please see Section [Building Containers]() about building containers. 

**Using the Docker Image** 

Assuming that the Docker image is named `sciml-bench`, the standard commands of the benchmark framework can be run by simply prefixing them with `sudo docker run` command. For example, the following are some examples: 

```sh 
sudo docker run --gpus all -v /tmp/bench_data:/root/bench_data sciml-bench list

sudo docker run --gpus all -v /tmp/bench_data:/root/bench_data sciml-bench download mnist

sudo docker run --gpus all -v /tmp/bench_data:/root/bench_data sciml-bench run MNIST_torch --output_dir=test-mnist
```

Here, the `-v` option here helps mounting the datasets on the host operating system. 

**Using the Singularity Image**

Assuming that the Singularity image is named `sciml-bench.sif`, the standard commands of the benchmark framework can be run by simply prefixing them with `singularity run` command. For example, the following are some examples: 

```sh
singularity run --nv sciml-bench.sif list
singularity run --nv sciml-bench.sif download mnist
singularity run --nv sciml-bench.sif run mnist_torch \
                --output_dir=test-mnist
```

Here, relevant folders are automatically accessible inside the host operating system. 

**Using with Third-Party Applications**

In addition to running the framework directly or through containers, it can also be passed on to third-party applications, such as `horovodrun`. For example, 

```sh
horovodrun -np 2 sciml-bench run mnist_torch --output_dir=demo_output_mnist_torch --monitor_interval=0.1 -b use_cuda false -b epochs 2
```

##  3.3 Outputs of the Runs 

Outputs of the benchmark runs are automatically logged into a sub-directory inside the main output directory, which is specified in the Configuration file. This is, by default, set to `~/sciml_bench_user/outputs/` but can be overridden.  The sub-directory name is same as the benchmark name, followed by date of run and type of run (training or inference). All the arguments used for a run are usually saved to the `arguments_used.yml` file in this  directory, subject to individual benchmarks deciding to do this. The sub-directory will also contain all log files capturing the outputs of the run. 

<div style="text-align: right"> ◼︎ </div>


