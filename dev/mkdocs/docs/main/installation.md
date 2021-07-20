# Installation
<br>
<br>

This part of the manual will guide you on installing and the benchmark suite.  There are three components that underpins benchmarking, namely, 

1. The core framework,
2. benchmarks, and 
3. datasets

The overall installation is divided into two parts, namely installation of the framework, and setting up the installation for usage.  Once configured, the framework and the suite can be used. These are covered in the following sections.  

<br>

## 1. Installing the Framework

It is worth noting that:

* **Supported Operating Systems**: Linux (and derivatives).

* **Software**: Python 3.9-upwards, pip, and nvidia-smi (if using CUDA/GPUs).

* **Benchmark-specific dependencies**: Please consult the benchmarks.


The framework and the benchmarks are primarily designed for large-scale systems (at least for training), and, as such,  windows or MacOS systems are not officially supported. However, we sound that the framework works on these non-Linux systems, and can be used primarily for inference purposes.

Although the core framework  does not heavily rely on machine learning frameworks, such as TensorFlow, PyTorch, or Apache MXNet, benchmarks may rely on one or more of these. As such,  supported Python versions are limited by these frameworks. In general, Python versions 3.8+ are generally supported, but we recommend version 3.9. We also rely on GPU-specific libraries, such as nvidia-smi for core capabilities. 

Benchmarks may rely on distributed learning, and subsequently, dependencies on distributed learning frameworks, such as TensorFlow Distributed, PyTorch Distributed Data Parallel (DDP) or Horovod, are inevitable. Although some of these can be installed through our framework, these frameworks may have complex dependencies and in some cases, OS-level installations. Hence, we recommend manual installation of these frameworks or we recommend using containers. 

<br>

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
   # print the version 
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


The `pip install` command installs framework-level packages within the Python environment you are using. Typing `which sciml-bench` can reveal you this directory. 


## 2 Configuring the Framework


The installation covers framework-level dependencies and does not cover documentation or potential customisations,  datasets,  benchmarks and dependencies of those benchmarks. As such, installation leaves the framework partially configured. Some of these settings are available inside the Configuration file. Please see the [Configuration](configuration.md) section for more details.

Setting up the framework involves installing benchmarks, their dependencies, and downloading relevant datasets. 


<br>

### 2.1 Installing Benchmarks

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

Each benchmark in the suite has its own dependencies, and these dependencies are automatically installed. We make our best efforts in ensuring that these dependencies do not conflict each other, but there can be cases of conflicts. In such cases, you may be required to install these manually (and may have to follow the on-screen instructions). Alternatively, you can use the container technology. 

The framework provides only a high-level summary for the installations, and a detailed log can be found inside the outputs folder. This is usually found in the `${HOME}/sciml_bench/install_logs` folder.

<br>

### 2.2 Downloading the Datasets

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


<br>


### 2.3 Installing Dependencies

Some of these benchmarks may have specific dependencies and these are, by default, installed. The framework makes the best effort for this and does not guarantee successful installations.  For example, [Horovod](https://horovod.ai/) is a framework used by a number of benchmarks, and may pose a challenge when installing automatically. In such cases, we recommend manual installation or using containers.  Please consult the FAQ section for specific discussions on Horovod. 



<br>

## 3 Setting up Container Images 

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


<div style="text-align: right"> ◼︎ </div>


