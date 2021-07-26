<img src="../resources/logo.png" alt="logo" width="500"/>

## 1. This Version

This is SciML-Bench Release Version: **1.1.0.b260721_0359.**
Please consult the RELEASE NOTES for more information. 
<br>

## 2. Synopsis

Benchmarking is a valuable tool in computational science that allows effective comparison of software and/or hardware systems.  With the rise of machine learning and AI, the need for benchmarking AI algorithms is  increasing. The SciMLBench is aimed at the ‘*AI for Science*’ research community.  

With an ever-growing number of machine learning models and algorithms, a range of scientific problems, and the proliferation of AI systems, the development of cutting-edge ML/AI algorithms requires a detailed understanding of all of these aspects as well as effective mechanisms for measuring their effectiveness.

The SciMLBench toolkit aims to provide such mechanisms. It is an open-source initiative and covers a range of scientific problems from various domains of science, including materials, the life sciences and environmental science, as well as particle physics and astronomy. The benchmarks are implemented in Python and rely on one or more machine learning frameworks, such as TensorFlow, PyTorch or SciKit-Learn.

The overarching purpose of this initiative is to support the ‘*AI for Science*’ community in the development of more powerful, robust and understandable machine learning solutions.


## 3. Benchmark Suite 
<br>

**3.1 Organisation of the Suite**

The suite has three key components, namely, 

1. *Benchmarks*: The benchmarks are machine learning applications performing a specific scientific task, written in Python. These are included as part of the distribution, and can be found inside the [benchmarks](../benchmarks) directory. In the scale of *micro-apps*, *mini-apps*, and *apps*, these are full-fledged applications. 

2. *Datasets*: Each benchmark in (1) relies on one or more datasets, for example for training and/or inferencing. These datasets are open, task- or domain-specific, and FAIR compliant. Most of these datasets being large, they are hosted separately,  on one of the servers (or mirrors), and are automatically or explicitly downloaded on demand. The framework (see (3)), supports manual downloading of these datasets. 

3. *Framework*:  The framework serves two purposes: first, at the user level, it facilitates an easier approach to benchmarking, logging and reporting of the results. Secondly, at the developer level, it provides a coherent API for unifying and simplifying the development of AI benchmarks. This can be found in [core](../core) directory. 

The source tree, which captures these aspects,  is organised as follows:

```bash

├── etc/                        <Contains essential extras>
│   ├── configs/                <Framework configurations>
│   └── recipes/                <Various container build recipes>
│   └── templates/              <Benchmark template(s)>
├── doc                         <Documentation files>
│   └── benchmarks/             <Documentation about benchmarks>
│   └── datasets/               <Documentation about datasets>
│   └── full/                   <Documentation in HTML format>
│   └── resources/              <Documentation resources>
├── benchmarks                  <Source tree for various benchmarks> 

```

We have annotated the purpose of each folder/directory within `<>`.  

**3.2 Features** 

A typical user-base for the benchmarking framework may include a number of user communities, such as system manufacturers and integrators (for assessing system performance), scientists (for developing new algorithms), and ML enthusiasts (for understanding the basics of various machine learning models and algorithms). It is a challenging task to design for and cover all these requirements in a single framework. Here, with SciMLBench, we have attempted to cover these requirements through the following set of features:

* Very flexible, customisable and lightweight framework,
* Powerful logging and monitoring capabilities, 
* Support for multiple machine learning frameworks (Tensorflow, PyTorch, and SciKit-Learn), 
* Simplified application programming interface (API), to support easier development of benchmarks, 
* Fully customisable installation, 
* Simplified use of framework encouraging a wide range of users, and
* Fully decoupled,  on-demand, and user-initiated data downloads. 



**3.3 Benchmarks and Datasets**

The number of [datasets](../datasets/summary.md) and [benchmarks](../benchmarks/summary.md) may vary with every release. A number of authors have contributed towards the development of the benchmarks,  and these can be see in the [Credits and Acknowledgment](../credits.md) file. If you are thinking of contributing towards the benchmarks or datasets, please see the [Contributing Datasets & Benchmarks](contributing.md) section.


## 4. Installation and Usage

Please consult the [Installation & Usage](usage.md) file for getting started. 



## 5. Citation 

Please cite this benchmark suite as follows:

```tex
@misc{scimlbench:2021,
    title  = {SciMLBench: A Benchmarking Suite for AI for Science},
    author = {Jeyan Thiyagalingam, Juri Papay, Kuangdai Leng, 
                    Samuel Jackson, Mallikarjun Shankar, 
                    Geoffrey Fox,  Tony Hey},
    url    = {https://github.com/stfc-sciml/sciml-bench},
    year   = {2021}
}
```


## 6. Acknowledgments

This benchmarking programme is supported by [1] Wave I of the UKRI Strategic Priorities Fund under the EPSRC grant (EP/T001569/1), particularly the *AI for Science* theme in that grant and the Alan Turing Institute, and [2] the Benchmarking for AI for Science at Exascale (BASE), EPSRC ExCALIBUR Phase I grant (EP/V001310/1). 

<div style="text-align: right">◼︎</div>

