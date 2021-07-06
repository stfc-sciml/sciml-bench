# Contributing Benchmarks and Datasets 



<br>

## 1. Introduction 

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
<br>

## 2.  The Configuration File 

As outlined in the main README file, a single benchmark in SciML-Bench includes two key elements, namely, (one or more) dataset(s), and a reference implementation in Python that solves a particular scientific problem.  We carefully curate the benchmarks (and relevant datasets) included in every release. 

The association between benchmarks and datasets is, as outlined before, is many-to-many. These relationships, relevant dependencies,  exact location of the datasets, and download methods (if any) are maintained in a single configuration file `etc/configs` folder. We recommend you to read the [Configuration File](configurations.md) section. 

During the development phase, the datasets do not have to  be deposited  in STFC servers.  Instead, they can be maintained at a location of your choice, for example a local directory. If datasets are locally stored, the `download` method can be left empty (or ignored), and instead the directory can be specified using the `--dataset_dir` option when launching the `sciml-bench run`.  By default, for all the benchmarks distributed by us, the datasets are maintained in the STFC servers (or its mirrors),  and you do not have to modify the download command. 

<br>

## 3. Developing Benchmarks

Each benchmark inside SciML-Bench must meet the minimum requirements outlined in the [API](./api.md) document. In addition to this documentation, we have also provided a number of examples in the suite, which can be found inside the `benchmarks/examples` folder.  Developing a benchmark involves two main aspects: 

1. Implementation of the benchmark, and 
1. Dataset(s) on which the implementation operates. 

We cover these two in the sections that follow this.

<br>

## 4. Contributing Benchmarks 

The scope of a single benchmark can be manyfold. However, in our case, a benchmark must meet the following criteria:

1. It must focus on a (ideally) practically significant scientific problem,
1. It should (ideally) use a machine learning technique to solve the problem in (1),
1. There is a clear and domain-specific metric to measure the outcome of the benchmarking exercise, and
1. There is a real or simulated dataset on which the benchmark operates. 

The contribution may come from any domain and can use any amount of data, but ideally on large (and open datasets). The aim here is to foster scientific development and growth than purely measuring performance (which, in fact, an interesting outcome).

The following are basic requirements when contributing a benchmark towards the SciML-Bench:

1. The code should be openly available without any form of restrictive licensing model. Ideally, we would promote MIT or BSD-style license models.
1. A Python-based implementation, strictly relying on Python 3+ and using one of the machine learning frameworks, like PyTorch, TensorFlow, MXNet and alike. Although we would encourage multiple implementations (covering each framework), it is not necessary. 
1. The implementation should at least focus on training or inference (or both). Depending on the focus of the benchmark, it has to include the implementations for  `sciml_bench_training()` and/or `sciml_bench_inference()`.
1. A clear documentation on the benchmark, outlining the primary domain where the problem comes from (such as material sciences, astronomy, particle physics, environmental sciences, earth sciences and alike), sub-domain of the problem, the intended learning task (estimation, classification etc), relevant datasets.
1. List of authors who contributed to the development / implementation of the benchmark.

There are best practices of how benchmarks should be implemented, but examples supplied with the suite are sufficient enough to learn these. Some of these are worth stating here:

1. Although there is nothing wrong with an implementation based on a single file, it is better to modularise the implementation, say using multiple files, as needed. 
1. Always use lower-case for benchmark names to avoid platform-specific issues. 
1. Consider the dependencies and libraries you are using. If the benchmark has complex dependencies and tends to be difficult for an average user to install, please reconsider this. In reality, if it is difficult to install, it is very unlikely that end users will use it. 


If you are contributing towards an existing case, follow the same set of rules, except that you provide this as one of the baselines. As such, the existing benchmarking code has to be modified.

<br>

# 5. Contributing Datsets 

Benchmarks operate on one or more datasets. As such, each benchmark should be accompanied with the relevant datasets.  The following are basic requirements when contributing a dataset towards the SciML-Bench:

1. The dataset(s) should be openly available without any form of restrictive licensing model. Ideally, we would promote MIT or BSD-style license models.
1. It should have a very clear and known data source (even if simulated) 
1. A clear documentation on the dataset, outlining the primary domain where the dataset comes from (such as material sciences, astronomy, particle physics, environmental sciences, earth sciences and alike), sub-domain of the dataset, data type (image, text, mixed, sound etc), and data size (in GB or TB).
1. A Python-based implementation to read or process the dataset.
1. List of authors who contributed to the dataset.

The dataset can be a pre-published one, but must be open. 