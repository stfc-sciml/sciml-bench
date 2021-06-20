# Contributing Benchmarks and Datasets 



# Table of Contents 



- [1. Introduction](#1-introduction)
  * [Before you start](#before-you-start)
- [2.  Registration of Benchmarks and Datasets](#2--registration-of-benchmarks-and-datasets)
- [3. Developing Benchmarks](#3-developing-benchmarks)
  * [3.1 Incorporating the benchmark implementation](#31-incorporating-the-benchmark-implementation)
  * [3.2 Registering a benchmark](#32-registering-a-benchmark)
  * [3.3 Handling benchmark-specific arguments](#33-handling-benchmark-specific-arguments)
  * [3.4  Logging and monitoring](#34--logging-and-monitoring)
  * [3.5 Benchmark dependencies and documentation](#35-benchmark-dependencies-and-documentation)





# 1. Introduction 

This page/file is intended to help you contribute to the SciML benchmarking initiatives. The contributions can be in different forms, from providing fully-fledged benchmarks to datasets to providing comments and suggestions to improve the benchmark suite. Although majority of this page is focused on enabling you to contribute towards new benchmarks, we have outlined the general ways to contribute to the SciML Bench suite in latter part of this document. 

##  Before you start

* Make sure that you are familiar with the notion of [benchmarking](../README.md), particularly, what benchmarks mean in the context of SciMLBench, and  [Installation & Usage](./installation_usage.md) of the SciMLBench Suite. 

* As a contributor, you can launch the command-line interface (CLI) in a different way to facilitate development, 
  i.e., from the repository folder,  

    ```sh
    python -m sciml_bench.core.command --help
    ```

where `sciml_bench.core.command` is the entry point of `sciml-bench`     after installation with `pip`. 

This way, any local changes to the repository will be visible to the CLI, avoiding unnecessary      re-installations.

​    

# 2.  Registration of Benchmarks and Datasets

As outlined in the [README](../README.md), a single benchmark in SciMLBench includes two key elements, namely, (one or more) dataset(s), and a reference implementation in Python that solves a particular scientific problem.  We carefully curate the benchmarks (and relevant datasets) included in every release. 



The association between benchmarks and datasets is, as outlined before, is many-to-many. These relationships, relevant dependencies,  exact location of the datasets, and download methods (if any) are maintained in a YAML-formatted file, namely, the [registration.yml](../sciml_bench/benchmarks/registration.yml) file inside the `sciml_bench` folder.  We show an example registration file below.



```yaml
# name of the dataset
MNIST:
    # approximate size
    size: "12 MB"
    # title of the dataset
    title: "The MNIST database of handwritten digits"
    # any information to display, such as credits, purposes and features
    info: "Added for demonstrating how to add a dataset to SciML-Benchmarks"
    # Credits 
    credits: "Author 1, Author 2, and Author 3"
    # download method
    # use either STANDARD_DOWNLOAD_STFC_HOST or commands taking $DATASET_DIR as input
    download_method: "wget -c https://www.yourserver.com/path/mnist.tar.bz2 -O - | tar -jx -C $DATASET_DIR"
    
# name of the benchmark
linreg_sklearn:
    # name of the used dataset (must be registered)
    dataset: MNIST
    # title of the dataset
    title: "Linear regression using sklearn"
    # any information to display, such as credits, purposes and features
    info: "Just a practice"
    # dependencies (just for information)
    dependencies: "sklearn"
```



As can be seen, entities (datasets or benchmarks) are marked with properties and the framework will automatically extract these properties and  relevant associations.  In the example above, the `liner_sklearn` benchmark relies on the `MNIST` dataset defined above. 



During the development phase, the datasets do not have to  be deposited  in central servers.  Instead, they can be maintained at a location of your choice, for example a local directory. If datasets are locally stored, the `download` method can be left empty (or ignored), and instead the directory can be specified using the `--dataset_dir` option when launching the `sciml-bench run`.  By default, for all the benchmarks distributed by us, the datasets are maintained in the STFC servers, and the download method will be `STANDARD_DOWNLOAD_STFC_HOST`. 

Using the example above, more  datasets can be added  using the same structure. The information provided by `size`, `title` and `info` will be used for displaying the benchmark properties. For data downloading, you need to provide the `sh` commands in `download_method`, taking `$DATASET_DIR` as the input, the destination directory passed by the user and created by the CLI.



# 3. Developing Benchmarks

