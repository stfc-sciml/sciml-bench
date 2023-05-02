
# SciML-Bench API

This document outlines the API and necessary examples for developing a benchmark for the SciML-Bench framework.

<br> 

## 1 Basic Notions 


To retain the framework flexible enough to include any complex benchmarks, SciML-Bench framework enforces very minimal API points on  the implementation of a benchmark. A benchmark can focus on training, or inference or both. As such, the API requirements are such that each benchmark implementation to have two function definitions: 

* `sciml_bench_training`, and
* `sciml_bench_inference`. 

In the absence of any of these methods, the framework will conclude that the benchmark does not support training or inference. The API primarily relies on two SciML-Bench-defined objects, `RuntimeIn`, and `RuntimeOut`.  The former carries in parameters from command line interface (CLI) or from configuration file to the benchmark while the latter carries out the results back to CLI or log files. 

Although a benchmark can be implemented on a single file, we recommend them to be modularised as necessary. Sources relating to a benchmark called `foo` should be located in a folder `sciml_bench/benchmarks/foo` and the entry points for the training and inference routines should be placed in a file named as `foo.py`. 

In a nutshell, adding a benchmark to SciML-Bench involves the following five steps:

1. Implementation of a benchmark,
2. Handling benchmark-specific arguments,
3. Logging and monitoring, and
4. Setting up the Configuration file,
5. Documentation.


<br>
<br>


## 2 Implementation of a Benchmark

In addition to naming your benchmarks fully in lower case, as mentioned in Section 1, the entry points, i.e. function definitions for  `sciml_bench_training` and `sciml_bench_inference` must be placed in a relevant file for the benchmark. We provide a template file in `sciml_bench/etc/templates`, named `example_benchmark.py`, which can be copied and modified as  desired from the source tree. Function definitions for both training and inference aspects should be modified (or deleted) as appropriate. 

### 2.1 Incoming and Outgoing Parameters 

Function signature for both the training and inference aspects are almost similar with two parameter types: `RuntimeIn` and `RuntimeOut`. The `RuntimeIn` parameter carries useful values into the benchmark function (training or inference) from CLI (mostly what user supplies) or from the Configuration file. The exact definitions of `RuntimeIn` and `RuntimeOut` can be found in `sciml_bench/core/runtime.py`.  The instances of these are named as `params_in` and `params_out`.


**RuntimeIn**

The **`RuntimeIn`** type carries the following attributes: 


* valid: Indicates whether this RuntimeIn is valid. This is usually set to False, if the configuration file is invalid or benchmark name is invalid. 
* error_msg: This contains the last error message when attempting to initialise the RuntimeIn variable.
* start_time: This marks the time when entering a function block (i.e. when the object has been instantiated)
* dataset_dir: Contains the dataset directory relevant to the benchmark. If none specified by the user, default value is set based on the benchmark and Configuration file.
* output_dir: Contains the output directory where the run / output logs are placed. If none specified by the user, default value is set based on the Configuration file.
* `bench_args`: These are benchmark-specific arguments, specified using the `-b` flag. 
* execution_mode: This specifies whether the benchmark is used on the inference or training mode. This can be specified by the user using the `--mode` flag during the `sciml_bench run` command. If the user does not specify the mode, it is set to training by default (provided that the entry point exists.)
* model: Contains the model file to be used for the inference purpose. This is only valid if the current execution mode is set to inference. User can specify the model file using the `--model` flag.


These can be access using the standard object notation, such as `params_in.dataset_dir` or `params_in.execution_mode`. Although all these attributes are available, not all of them are often used. In majority of the cases, `start_time`, `dataset_dir`, `output_dir`, `bench_args` and `model` are useful. 


**RuntimeOut**

The **`RuntimeOut`** mostly concerns about the logs, and loggers used to drive these logs. The SciML-Bench relies on a custom developed multi-level logging system that accounts not only for the hierarchical logging, but also logging on different hosts, nodes and processes, which is often demanded when using distributed training using multi-GPU or multi-node systems. With multiple nodes, each with multiple GPUs available, it soon becomes very confusing to differentiate which GPU resulted in which log. For this purpose, we use their `rank` information, which is usually supported / provided by distributed learning frameworks, such as Horovod or TensorFlowDistributed or PyTorch DistributedDataParallel (DDP).  To avoid confusion, there are two levels of rank information: `rank` and `local_rank`. For example, if there are four compute nodes each with four GPUs, there will be 16 workers, with each worker being assigned a `rank` [0, 15], and every worker will also have a `local_rank` [0, 3]. It carries the following attributes: 


* log.console: This is a handle to the multi-level logger on the global host / root device (i.e. device with `rank`=0)
* log.host: multi-level logger on host (i.e. `local_rank`=0)
* log.device: multi-level logger on device (i.e. `rank` can be anything)
* system: a set of system monitors

Unlike `RuntimeIn` or its instance `params_in`, to use `RuntimeOut` or its instance `params_out`, it must be activated. The `activate` function has the following arguments: `rank`, `local_rank`,`activate_log_on_host`,
`activate_log_on_device`, and `console_on_screen`. Here, the rank information is essential for enabling logging from various processes. These  `rank` or `local_rank` information are usually available from the distributed learning environment used by the benchmark. A typical call for activating the monitor will look like 

```python
params_out.activate(rank=0, local_rank=0, activate_log_on_host=True,
                      activate_log_on_device=True, console_on_screen=True)
```

For a non-distributed benchmark, both `rank` and `local_rank` can be set to `0`, as in `rank=0`, `local_rank=0`. 
, and by setting the `activate_log_on_host=False`. There is nothing wrong in setting this to `True` as the log on `host:0` and `device:0` will be the same as that on console except for some small differences in time measurements. Usually, the boolean flags do not need to be set explicitly, unless to turn them off. This activation is tightly coupled to logging system used by the framework. The framework supplies an auxiliary set of  APIs for producing attractive runtime logs and system reports, supporting distributed learning to facilitate minimum coding from benchmark developers. This is discussed in the next section. 

<br>
<br>


## 3 Monitor Activation & Logging


### 3.1 Activation 
As discussed in Section 2.1, activating the instance of RuntimeOut is essential for enabling the logging process, which inherently relies on the rank information, which are crucial for differentiating the logs produced by different GPUs from different nodes. Each process carries the following information: `rank`, `local_rank`, and a set of boolean flags for `root`, `host` and `device`. Consider a system with three nodes each with two GPUs (in other words,  three hosts with two GPUs each). 

* **Table 1: DLE of 3 hosts ✕ 2 GPUs per host**

| Process on | `rank` | `local_rank` | root | host | device 
| ---|:---: | :---: | --- | --- | --- |
|  Host 1, GPU 1| 0 | 0 | True | True | True |
|  Host 1, GPU 2| 1 | 1 | False | False | True |
|  Host 2, GPU 1| 2 | 0 | False | True | True |
|  Host 2, GPU 2| 3 | 1 | False | False | True |
|  Host 3, GPU 1| 4 | 0 | False | True | True |
|  Host 3, GPU 2| 5 | 1 | False | False | True |


As we will see in latter sections, developing benchmarks relying on distributed training requires the support of a distributed learning/training environment. The rank information are usually supplied by them.  With the notions of `root`, `host` and `device` being cleared, it is worth noting that, `RuntimeOut.log.console` refers to the logger on root, and will always be activated. This can be made available to the screen by setting `console_on_screen=True`. 
Developers can enable  `RuntimeOut.log.host` or `RuntimeOut.log.device` as appropriate. The latter refers to the actual compute device and enabling logging at the device level provides very fine-grained granularity. 

In a  non-distributed training setting, there is no need to activate the host and the device loggers. Here, all the system monitors will be activated. Those on host will monitor the total system usage on that node or machine, including all existing CPUs and GPUs. However, this monitoring does not differentiate whether  they are actually used by the running benchmark or not, and we hope to improve this soon. In contrast, those on device will monitor all types of resources  used by this running instance. In other words, those on host monitor the machine while  those on device monitor the processes.

### 3.2 Logging 

Logging within the SciML-Bench is event triggered. The contributors need to add logging routines wherever necessary. Once the monitor is activated, a handle for the suitable logger can be obtained. For instance  

```python
console = params_out.log.console
```

provides a handle for the console logger (which is enabled by default). The logger provides the following top-level methods for enabling logging: `begin()`, `ended()` and `message()`. Calling these functions results in the method to be invoked on console, host and device loggers at the same time. The API also provides another function `subproc()` to be used with the `with` clause, which can be used to simplify things. Consider the following logging example: 

```python
console = params_out.log.console
console.begin('foo')
# do something
console.message('bar')  
console.ended('foo')
```

This is equivalent to the following, simplified version: 

```python
with logger.subproc('foo'):
    # do something
    logger.message('bar')  
```

The logging can be nested as needed, enabling the framework to produce an elegant log reports, like the following: 

```sh
    <BEGIN> Running benchmark lr_sklearn
    ....<BEGIN> Randomizing input data
    ....<ENDED> Randomizing input data [ELAPSED = 0.000105 sec]
    ....<BEGIN> Getting benchmark-specific argument
    ........<MESSG> normalize = False
    ....<ENDED> Getting benchmark-specific argument [ELAPSED = 0.000078 sec]
    ....<BEGIN> Training
    ........<BEGIN> Fitting data
    ........<ENDED> Fitting data [ELAPSED = 0.001196 sec]
    ........<BEGIN> Saving coefficients and intercepts
    ........<ENDED> Saving coefficients and intercepts [ELAPSED = 0.002882 sec]
    ....<ENDED> Training [ELAPSED = 0.004370 sec]
    ....<BEGIN> Computing score
    ........<MESSG> score = 1.0
    ....<ENDED> Computing score [ELAPSED = 0.000685 sec]
    ....<BEGIN> Making predictions
    ....<ENDED> Making predictions [ELAPSED = 0.000095 sec]
    <ENDED> Running benchmark lr_sklearn [ELAPSED = 0.005910 sec]
```

This is usually achieved using the calling the `subproc` method of the appropriate logger. The following code produces the log output above: 

```python
  # libs from sciml_bench
  from sciml_bench.core.runtime import RuntimeIn, RuntimeOut
  
  # libs required by implementation
  import numpy as np
  from sklearn.linear_model import LinearRegression
  
    def sciml_bench_training(params_in: RuntimeIn, params_out: RuntimeOut):

        params_out.activate(rank=0, local_rank=0)
    
        # begin top-level
        console = params_out.log.console
        console.begin('Running benchmark lr_sklearn')
    
        # input
        with console.subproc('Creating some input data'):
            X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
            y = np.dot(X, np.array([1, 2])) + 3
    
        # get benchmark-specific arguments
        with console.subproc('Getting benchmark-specific arguments'):
            normalize = params_in.bench_args.try_get('normalize', default=False)
            console.message(f'normalize = {normalize}')
    
        # fit
        console.begin('Training')
        with console.subproc('Fitting data'):
            reg = LinearRegression(normalize=normalize).fit(X, y)
    
        # save coefficients and intercepts
        with console.subproc('Saving coefficients and intercepts'):
            np.savetxt(params_in.output_dir / 'coefficients.txt', reg.coef_)
            np.savetxt(params_in.output_dir / 'intercepts.txt', [reg.intercept_])
        console.ended('Training')
    
        # score
        with console.subproc('Computing score'):
            console.message(f'score = {reg.score(X, y)}')
    
        # predict
        with console.subproc('Making predictions'):
            reg.predict(np.array([[3, 5]]))
    
        # end top_level
        console.ended('Running benchmark lr_sklearn')
``` 
    
   
Inevitably, these logging may make the code less readable and longer.


## 3.3 Event Stamping

In addition to the logging, system usage is sampled by the framework at a regular interval of one second (which can be modified  through the `--monitor_interval`  flag). The system usage, usually, will not have any log-specific messages To facilitate correlating the logs with system usage, events can be stamped, This can be achieved by calling the `stamp_event()` method with the appropriate message. For instance, the following code will stamp *<<LR.fit <<begin>>* and *LR.fit <<end>>* stamps into the system usage log. 


```python
    with console.subproc('Fitting data'):
          params_out.system.stamp_event('LR.fit <<begin>>')
          reg = LinearRegression(normalize=normalize).fit(X, y)
          time.sleep(10)  # sleep, as if it is doing fit()
          params_out.system.stamp_event('LR.fit <<end>>'')
```

<br>
<br>





## 4 Handling Benchmark-Specific Arguments
<br>

Benchmark-specific arguments can be specified using the  `-b` option for the `sciml-bench run` command. Internally, these arguments are passed to `sciml_bench_training()` and `sciml_bench_inference()` routines through the `RuntimeIn` instance, namely, `params_in`. These can be accessed by `params_in.bench_args`. These benchmark-specific arguments are specified through the `-b` flag followed by key-value pair. If there are are more than one key-value pair, each of them are expected to be specified using the `-b` flag. These can extracted through the `try_get` method as in the example below:


```python
# hyperparameters
batch_size = params_in.bench_args.try_get('batch_size', default=64)
epochs = params_in.bench_args.try_get('epochs', default=2)
```

The `try_get()` method will return the default value if the key is missing. The value presented in `params_in.bench_args` will be returned with an appropriate typecasting (based on the type of `default` value), failing which, an error will be raised.  It is also possible to extract these arguments in bulk by using the `try_get_dict` method by passing in a Python dictionary of arguments, as shown in the example below: 


```python
# default arguments
default_args = {
    # torch env
    'seed': 1234, 'use_cuda': True,
    # network parameters
    'n_filters': 32, 'n_unit_hidden': 128,
    # hyperparameters
    'epochs': 2, 'batch_size': 64, 'loss_func': 'CrossEntropyLoss',
    'optimizer_name': 'Adam', 'lr': 0.001,
    'batch_size_test': 2000,  # use a large one for validation
    'batch_log_interval': 100,  # interval to log batch loop
    # workflow control
    'load_weights_file': '',  # do training if this is empty
}
# replace default_args with bench_args
# Note: try_get_dict() provides a way to do try_get() collectively.
args = params_in.bench_args.try_get_dict(default_args=default_args)
```

In this case, the `try_get_dict()` method will perform `try_get()` on each key-value pair
in `default_args`. When specifying flags, such as whether to normalise or not, the `-b` flag accepts all possible alternatives, including  `true/false`, `t/f`, `yes/no`, `y/n`, and `1/0`,  treating indifferently and without being sensitive to their cases. For example, the following are the same: 

```sh
python -m sciml_bench.core.command run lr_sklearn --output_dir=lr_out -b normalize YES
```
and 

```sh
python -m sciml_bench.core.command run lr_sklearn --output_dir=lr_out -b normalize 1
```


Finally, where there are  too many arguments, the  default key-value pairs can be read from a text file a well. 




## 5 Setting up the Configuration File
<br> 

Each benchmark / dataset should have an entry in the configuration file. Although this configuration file is provided with the SciML-Bench release, you can modify the configuration file for developmental and testing purposes. Please see the documentation around Configuration File for more information. 


## 6 Documenting Benchmarks and Datasets
<br>

Each benchmark should come with a detailed documentation outlining the challenge and the scope. It should provide the following information:


 * Main scientific domain where the benchmark problem originates from
 * Sub scientific domain of the benchmark 
 * What is the scope of the learning Task. Is it classification or regression and so on.
 * Relevant datasets to these benchmarks (See the documentation around datasets).
 * Authors of the benchmark. 

In addition to the detailed description, each benchmark should also provide a short description not exceeding 120 words. This short description is used for the `info` command. 


## 8 Example Implementation - A Walk-through 

For instance, assume that we would like to create an example based on sklearn-based linear regression as in [here](https://bityl.co/7OON), and assume that we would like to name it as *lr_sklearn*. We will also assume that this benchmark will support both training and inference. Thus, the benchmark must have entry points for `sciml_bench_training` and `sciml_bench_inference`. 


We can start off by copying the template and renaming the file. 

```sh
# copy and rename the core file
cp -r sciml_bench/etc/templates/example_benchmark.py sciml_bench/benchmarks/lr_sklearn
mv sciml_bench/benchmarks/lr_sklearn/example_benchmark.py sciml_bench/benchmarks/lr_sklearn/lr_sklearn.py
```

Next, implement the `sciml_bench_training` as below:

```python
# libs from sciml_bench
from sciml_bench.core.runtime import RuntimeIn, RuntimeOut

# libs required by implementation
import numpy as np
from sklearn.linear_model import LinearRegression


def sciml_bench_training(params_in: RuntimeIn, params_out: RuntimeOut):

        params_out.activate(rank=0, local_rank=0)
    
        # begin top-level
        console = params_out.log.console
        console.begin('Running benchmark lr_sklearn')
    
        # input
        with console.subproc('Creating some input data'):
            X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
            y = np.dot(X, np.array([1, 2])) + 3
    
        # get benchmark-specific arguments
        with console.subproc('Getting benchmark-specific arguments'):
            normalize = params_in.bench_args.try_get('normalize', default=False)
            console.message(f'normalize = {normalize}')
    
        # fit
        console.begin('Training')
        with console.subproc('Fitting data'):
            reg = LinearRegression(normalize=normalize).fit(X, y)
    
        # save coefficients and intercepts
        with console.subproc('Saving coefficients and intercepts'):
            np.savetxt(params_in.output_dir / 'coefficients.txt', reg.coef_)
            np.savetxt(params_in.output_dir / 'intercepts.txt', [reg.intercept_])
        console.ended('Training')
    
        # score
        with console.subproc('Computing score'):
            console.message(f'score = {reg.score(X, y)}')
    
        # end top_level
        console.ended('Running benchmark lr_sklearn')
```

In the above implementation, we save the model coefficients and intercept as outputs. In general,  benchmark-specific outputs are not managed by SciML-Bench, as different benchmarks may have different requirements for output. Common outputs of this kind include used hyperparameters, model weights, metrics and training history.

Next, we will implement the `sciml_bench_inference`. For this, we will assume that the user specifies the model files. Here, there are two model files to be used `coefficients.txt` and `intercepts.txt`. This can be specified using multiple `--model` argument.  Here, it has to be specified as: `--model /tmp/coefficients.txt --model /tmp/intercepts.txt`. Not all inference routines rely on multiple models, but this is an example and these must be handled in the inference routine as required. It is worth noting that models are automatically named based on the file names of the models. The inference can be performed in a number of different ways: on all files from the dataset directory for inference (which has no default value) or through a benchmark-specific argument. Here, we will assume that we would like to do the inference on a number of files situated inside a dataset directory that the user specifies.  In a nutshell, it can be assumed that the user performed the inference as follows: 


```sh
sciml-bench run --mode inference --model /tmp/coefficients.txt --model /tmp/intercepts.txt -dataset_dir /tmp/inference lr_sklearn 
```

Instead of using the `dataset_dir` argument, the benchmark-specific argument can also be used, say using a file name or similar. Now, we will load these models and perform inference.  For clarity, we have avoided extensive logging:


```python


def sciml_bench_inference(params_in: RuntimeIn, params_out: RuntimeOut):

        params_out.activate(rank=0, local_rank=0)
    
        output_file = params_in.output_dir  / 'lr_inference.log'

        # begin top-level
        console = params_out.log.console
        console.begin('Running benchmark lr_sklearn on inference mode')
    
        # input
        with console.subproc('Building the model'):
            model = linear_model.LinearRegression()
            model.coef_ = np.loadtxt(params_in.model['coefficients])
            model.intercept_ = np.loadtxt(params_in.model['intercepts'])

        # Do prediction for each data file 
        p = params_in.dataset_dir.glob('**/*')
        files = [x for x in p if x.is_file()]
        with open(output_file, 'w') as out_f:
        for f in files:
            data = np.loadtxt(f)
            with console.subproc('Fitting data'):
                y = model.predict(data)
                out_f.write(f'{f}\t{y}\n')

        
        # end top_level
        console.ended('Running benchmark lr_sklearn on inference mode')
```

