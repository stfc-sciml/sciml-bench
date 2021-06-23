# Contributing Benchmarks and Datasets 


# Table of Contents 


# 1. Introduction 

This page/file is intended to help you contribute to the SciML benchmarking initiatives. The contributions can be in different forms, from providing fully-fledged benchmarks to datasets to providing comments and suggestions to improve the benchmark suite. Although majority of this page is focused on enabling you to contribute towards new benchmarks, we have outlined the general ways to contribute to the SciML Bench suite in latter part of this document. 

##  Setting up the Environment

* Make sure that you are familiar with the notion of benchmarking, particularly, what benchmarks mean in the context of SciML-Bench, and the installation of the suite. 

* As a contributor, you can launch the command-line interface (CLI) in a number of ways to facilitate the development,  For example, from the repository folder,  

    ```sh
    python -m sciml_bench.core.command --help
    ```

where `sciml_bench.core.command` is the entry point of `sciml-bench`  after installation with `pip`. 

This way, any local changes to the repository will be visible to the CLI, avoiding unnecessary      re-installations.

​    

# 2.  Configuration File 

As outlined in the main README file, a single benchmark in SciML-Bench includes two key elements, namely, (one or more) dataset(s), and a reference implementation in Python that solves a particular scientific problem.  We carefully curate the benchmarks (and relevant datasets) included in every release. 


The association between benchmarks and datasets is, as outlined before, is many-to-many. These relationships, relevant dependencies,  exact location of the datasets, and download methods (if any) are maintained in a single configuration file `etc/configs` folder. We recommend you to read the [Configuration File](./configurations.md) section. 


During the development phase, the datasets do not have to  be deposited  in STFC servers.  Instead, they can be maintained at a location of your choice, for example a local directory. If datasets are locally stored, the `download` method can be left empty (or ignored), and instead the directory can be specified using the `--dataset_dir` option when launching the `sciml-bench run`.  By default, for all the benchmarks distributed by us, the datasets are maintained in the STFC servers (or its mirrors),  and you do not have to modify the download command. 


# 3. Developing Benchmarks

Each benchmark inside SciML-Bench must meet the minimum requirements outlined in the [API](./benchmark_API.md) document. In addition to this documentation, we have also provided a number of examples in the suite, which can be found inside the `benchmarks/examples` folder. 


# 4. Contributing Datasets 

# 5. Contributing Benchmarks 