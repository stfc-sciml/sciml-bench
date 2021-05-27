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



To retain the framework flexible enough to include any complex problems,
SciMLBench suite does not enforce any specific structure on  the implementation of the benchmark. 
Each benchmark is incorporated into SciMLBench as a module using a well-defined,  function definition, which serves as a unique  entry-point, namely, the `sciml_bench_run()`function.  This function connects the benchmark to the command line interface (`sciml-bench run`),  as shown in the following figure:

<img src="./resources/entry.png" alt="entry" width="800"/>

When the user do `$ sciml-bench run some_benchmark`, the CLI initializes
two objects, `smlb_in` and `smlb_out`, and pass them to the benchmark module through `sciml_bench_run()`, which looks like the following:

```python
def sciml_bench_run(smlb_in: RuntimeIn, smlb_out: RuntimeOut):
    """
    Main entry of `sciml_bench run` for a benchmark instance

    :param smlb_in: runtime input of `sciml_bench run`, useful components:
        * smlb_in.start_time: start time of running as UTC-datetime
        * smlb_in.dataset_dir: dataset directory
        * smlb_in.output_dir: output directory
        * smlb_in.bench_args: benchmark-specific arguments
    :param smlb_out: runtime output of `sciml_bench run`, useful components:
        * smlb_out.log.console: multi-level logger on root (rank=0)
        * smlb_out.log.host: multi-level logger on host (local_rank=0)
        * smlb_out.log.device: multi-level logger on device (rank=any)
        * smlb_out.system: a set of system monitors
    """
    print('Hello world! I am a template.')
```

The`smlb_in` object contains the CLI-managed input passed to the benchmark, 
including the dataset and output directories and the benchmark-specific arguments specified by `-b` in `sciml-bench run`.  The `smlb_out` object, on the other hand,  contains the CLI-managed output from the benchmark, including the built-in loggers and system monitors. Because the distributed learning environment (DLE) must be initialized by  the benchmark itself, the benchmark must activate `smlb_out` with its DLE before using it. We will demonstrate the use of `smlb_in` and `smlb_out`
with examples.


In brief, adding a benchmark to SciMLBench includes five steps:

1. Benchmark implementation
2. Registering a benchmark;
3. Handling benchmark-specific arguments;
4. Logging and monitoring; and
5. Specifying benchmark dependencies and documentation.

We will demonstrate the whole process using a simple example: 
linear regression with `sklearn`. Two benchmark examples using the `MNIST` dataset,
`MNIST_tf_keras` and `MNIST_torch` (with horovod-based distributed learning), 
are also provided for demonstration, which include more complicated
usage of `smlb_in` and `smlb_out`.




##  3.1 Incorporating the benchmark implementation

For a benchmark named `bar`, the entry-point function `sciml_bench_run()` must be
implemented in the Python file `sciml_bench/benchmarks/bar/bar.py`.
The folder `sciml_bench/benchmarks/bar/` can contain many other source and 
resource files for the implementation. 
There is a template folder under `sciml_bench/benchmarks/` for quick start.

To kick-off our linear regression example, as named `linreg_sklearn`, we can do

```sh
# copy the folder
cp -r sciml_bench/benchmarks/template sciml_bench/benchmarks/linreg_sklearn

# rename the file
mv sciml_bench/benchmarks/linreg_sklearn/template.py sciml_bench/benchmarks/linreg_sklearn/linreg_sklearn.py
```

Next, we can add the implementation to the entry-point function `sciml_bench_run()`.
Here we just borrow a simple example from the [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) of `sklearn`:


```python
# libs from sciml_bench
from sciml_bench.core.runtime import RuntimeIn, RuntimeOut

# libs required by implementation
import numpy as np
from sklearn.linear_model import LinearRegression


def sciml_bench_run(smlb_in: RuntimeIn, smlb_out: RuntimeOut):
    """ Main entry of `sciml_bench run` for a benchmark instance """
    # input
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    # fit
    reg = LinearRegression().fit(X, y)
    # save coefficients and intercept
    np.savetxt(smlb_in.output_dir / 'coefficients.txt', reg.coef_)
    np.savetxt(smlb_in.output_dir / 'intercept.txt', [reg.intercept_])
    # score
    reg.score(X, y)
    # predict
    reg.predict(np.array([[3, 5]]))
```

In the above implementation, we save the model coefficients and intercept as an output.
Such benchmark-specific outputs are not managed by SciMLBench, 
as different benchmarks may have different requirements for output.
Common outputs of this kind include used hyperparameters,
model weights, metrics and training history.



## 3.2 Registering a benchmark

To make a benchmark visible to the CLI, as stated in the previous section, each benchmark  must be registered  within the `registration.yml` file  as  shown above.   Registered benchmarks can be listed via the  `list` command as follow:

```sh
python -m sciml_bench.core.command list --verify
```

To run the benchmark:

```sh
python -m sciml_bench.core.command run linreg_sklearn --output_dir=linreg_out
```

Once the run is completed,  the output folder, `linreg_out`,  created within the current directory, will contain the outputs emitted by the benchmark. In this case, they will be `coefficients.txt` and `intercept.txt` .

Benchmarks, inevitably, will have dependencies. We will discuss how these dependencies can be handled in an automatic manner further below. If the dependencies are not satisfied, the `list` command display error messages for each of the of benchmark. For example,  the `linreg_sklearn` benchmark relies on sklearn. However, if the sklearn is not available in the system / environment, a  `Modules verified: False` will appears under `linreg_sklearn`, These dependencies can be installed using standard environment commands, such as `pip install sklearn`. 



## 3.3 Handling benchmark-specific arguments

Benchmark-specific arguments can be specified using the  `-b` option for the `sciml-bench run` command. Internally, these arguments are passed to `sciml_bench_run()` by `smlb_in.bench_args`,  which is an instance of the `SafeDict` class defined in  [utils.py](../sciml_bench/core/utils.py).

To get a single argument with default value, for example, in  `MNIST_tf_keras` ([MNIST_tf_keras.py](../sciml_bench/benchmarks/MNIST_tf_keras/MNIST_tf_keras.py)):

```python
# hyperparameters
batch_size = smlb_in.bench_args.try_get('batch_size', default=64)
epochs = smlb_in.bench_args.try_get('epochs', default=2)
```

The `try_get()` method will return the default value if the key is missing. The value presented in `smlb_in.bench_args` will be returned with a typecasting.  However, an error will be raised if the key exists but typecasting fails (`str` to the type of `default`). 

To get multiple arguments in a collective manner, for example, in `MNIST_torch` ([MNIST_torch.py](../sciml_bench/benchmarks/MNIST_torch/MNIST_torch.py)):

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
args = smlb_in.bench_args.try_get_dict(default_args=default_args)
```

In this case, the `try_get_dict()` method will perform `try_get()` on each key-value pair
in `default_args`, so the output will have the same keys as `default_args` but
with some values replaced by those presented in `smlb_in.bench_args`.


For the `linreg_sklearn` example, let us expose `normalize` as an argument:

```python
# libs from sciml_bench
from sciml_bench.core.runtime import RuntimeIn, RuntimeOut

# libs required by implementation
import numpy as np
from sklearn.linear_model import LinearRegression


def sciml_bench_run(smlb_in: RuntimeIn, smlb_out: RuntimeOut):
    """ Main entry of `sciml_bench run` for a benchmark instance """
    # input
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.dot(X, np.array([1, 2])) + 3
    # get benchmark-specific arguments
    normalize = smlb_in.bench_args.try_get('normalize', default=False)
    # fit
    reg = LinearRegression(normalize=normalize).fit(X, y)
    # save coefficients and intercept
    np.savetxt(smlb_in.output_dir / 'coefficients.txt', reg.coef_)
    np.savetxt(smlb_in.output_dir / 'intercept.txt', [reg.intercept_])
    # score
    reg.score(X, y)
    # predict
    reg.predict(np.array([[3, 5]]))
```

Now the user can run `linreg_sklearn` with `-b normalize true/false` option.  The `-b` flag accepts the alternatives, such as `true/false`, `t/f`, `yes/no`, `y/n`, and `1/0`, and these are treated case-insensitively.

```sh
python -m sciml_bench.core.command run linreg_sklearn --output_dir=linreg_out -b normalize YES
```

An error will be raised if an  invalid argument is speciffied, for example,  `-b normalize -1`.

**Additional Points**:

* Where there are  too many arguments, the  default key-value pairs can be read from a configuration file; such configuration files should be packed with the benchmark instead of the dataset.

* If you have written your own parser in the original implementation, such as using `argparse`, 
  you can get the `args` from the user by `-b args "--foo --bar=1"` and do
       

    ```python
    # assume parser is ready
    args = parser.parse_args(smlb_in.bench_args.try_get('args', default='').split(' '))
    ```

* The benchmark-specific arguments can be used to control job workflow, such as
  `load_weights_file` in `MNIST_torch` (skip training and load weights from a file).

* At least one argument should be accepted to limit the wall time; in many cases, this would
  be the number of epochs.

* Usually, it is good practice to save all the actually used arguments as an output.



## 3.4  Logging and monitoring

The contributors are provided with handy tools to produce runtime logs 
and system reports in a distributed learning environment (DLE) with minimum coding.
The loggers and system monitors are initialized by the CLI 
and passed to the benchmark by `smlb_out`. It contains the following components:

* `smlb_out.log.console`: a multi-level logger activated only if this process is on root (`rank`=0);
* `smlb_out.log.host`: a multi-level logger activated only if this process is on a host (`local_rank`=0);
* `smlb_out.log.device`: a multi-level logger activated on every process or device (`rank`=any);
* `smlb_out.system`: a set of system monitors, 
  some for the node (or machine) on host (`local_rank`=0) and the others
  for the process on device (`rank`=any).

To quickly explain the DLE, let us assume that we have three hosts (nodes), each has two GPUs (devices), the DLE can be summarized in the following table (note that the order of `rank` and `local_rank` may vary):

* **Table 1: DLE of 3 hosts ✕ 2 GPUs per host**

  | Process on | `rank` | `local_rank` | root | host | device 
  | --- | :---: | :---: | --- | --- | --- |
  | GPU 1, Host 1 | 0 | 0 | True | True | True |
  | GPU 2, Host 1 | 1 | 1 | False | False | True |
  | GPU 1, Host 2 | 2 | 0 | False | True | True |
  | GPU 2, Host 2 | 3 | 1 | False | False | True |
  | GPU 1, Host 3 | 4 | 0 | False | True | True |
  | GPU 2, Host 3 | 5 | 1 | False | False | True |

For a non-distributed learning, we have one process on one device. Thus,  `rank`=`local_rank`=0.


* **Activating `smlb_out` with DLE**

  Because the DLE must be initialized by the benchmark (e.g., different benchmarks
  may use different libraries for parallelization), `smlb_out` must be activated
  with the DLE. To use the built-in loggers and monitors,
  the following line must be added to `sciml_bench_run()` after
  the initialization of the DLE (as soon as `rank` and `local_rank` are known):

  ```python
  smlb_out.activate(rank=rank, local_rank=local_rank,
                    activate_log_on_host=activate_log_on_host,
                    activate_log_on_device=activate_log_on_device, 
                    console_on_screen=console_on_screen)
  ```

  `smlb_out.log.console`, the logger on root, will always be activated, and the contributors
  can choose whether to synchronize this log on screen (stdout) during runtime by passing 
  `console_on_screen`. 

  Based on the benchmark implementation, the contributors can decide whether to activate `smlb_out.log.host` and `smlb_out.log.device`.  For example, in `MNIST_torch`, we only activate `smlb_out.log.device` to use it inside the training loop (where task distribution actually occurs).
  For non-distributed learning, there is no need to activate the host and the device loggers. Here, all the system monitors will be activated. Those on host will monitor the gross system usage on that node or machine, including all existing CPUs and GPUs no matter they are used by this running instance (the `sciml-bench run` process) or not.  In contrast, those on device will monitor all types of resources  used by this running instance. In other words,  those on host monitor the machine while  those on device monitor the processes.

* **Using the multi-level loggers**

  Logging within the SciMLBench is event triggered. The contributors need to add logging routines wherever necessary.  Taking `linreg_sklearn` as an example, 

  ```python
  # libs from sciml_bench
  from sciml_bench.core.runtime import RuntimeIn, RuntimeOut
  
  # libs required by implementation
  import numpy as np
  from sklearn.linear_model import LinearRegression
  ```


    def sciml_bench_run(smlb_in: RuntimeIn, smlb_out: RuntimeOut):
        """ Main entry of `sciml_bench run` for a benchmark instance """
        # activate smlb_out
        smlb_out.activate(rank=0, local_rank=0)
    
        # begin top-level
        console = smlb_out.log.console
        console.begin('Running benchmark linreg_sklearn')
    
        # input
        with console.subproc('Creating some input data'):
            X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
            y = np.dot(X, np.array([1, 2])) + 3
    
        # get benchmark-specific arguments
        with console.subproc('Getting benchmark-specific arguments'):
            normalize = smlb_in.bench_args.try_get('normalize', default=False)
            console.message(f'normalize = {normalize}')
    
        # fit
        console.begin('Training')
        with console.subproc('Fitting data'):
            reg = LinearRegression(normalize=normalize).fit(X, y)
    
        # save coefficients and intercept
        with console.subproc('Saving coefficients and intercept'):
            np.savetxt(smlb_in.output_dir / 'coefficients.txt', reg.coef_)
            np.savetxt(smlb_in.output_dir / 'intercept.txt', [reg.intercept_])
        console.ended('Training')
    
        # score
        with console.subproc('Computing score'):
            console.message(f'score = {reg.score(X, y)}')
    
        # predict
        with console.subproc('Making predictions'):
            reg.predict(np.array([[3, 5]]))
    
        # end top_level
        console.ended('Running benchmark linreg_sklearn')
    ``` 
    
    Running the benchmark, you will see the following messages on screen and in
    `linreg_out/sciml_bench_run_logs/console.log`:
    
    ```
    <BEGIN> Running benchmark linreg_sklearn
    ....<BEGIN> Randomizing input data
    ....<ENDED> Randomizing input data [ELAPSED = 0.000105 sec]
    ....<BEGIN> Getting benchmark-specific argument
    ........<MESSG> normalize = False
    ....<ENDED> Getting benchmark-specific argument [ELAPSED = 0.000078 sec]
    ....<BEGIN> Training
    ........<BEGIN> Fitting data
    ........<ENDED> Fitting data [ELAPSED = 0.001196 sec]
    ........<BEGIN> Saving coefficients and intercept
    ........<ENDED> Saving coefficients and intercept [ELAPSED = 0.002882 sec]
    ....<ENDED> Training [ELAPSED = 0.004370 sec]
    ....<BEGIN> Computing score
    ........<MESSG> score = 1.0
    ....<ENDED> Computing score [ELAPSED = 0.000685 sec]
    ....<BEGIN> Making predictions
    ....<ENDED> Making predictions [ELAPSED = 0.000095 sec]
    <ENDED> Running benchmark linreg_sklearn [ELAPSED = 0.005910 sec]
    ```
    
    Here are the explanations:
    * By default, the device and host loggers are unactivated 
    (`activate_log_on_host=False` and `activate_log_on_device=False`), so the line
    `smlb_out.activate(rank=0, local_rank=0)` will only activate log.console.
    * You can see from the result why our loggers are called multi-level loggers.
    The resulting log can reflect the logical levels of sub-processes by automatic indentation; 
    the logical level is increased by 1 upon calling `begin()` and decreased
    by 1 upon calling `ended()`. Note that the event string can be omitted when calling `ended()`.
    * For a small section of code to be placed between `begin()` and `ended()`,
    you can use the `with subproc()` statement to make the code more concise:
        
        ```python
        with logger.subproc('foo'):
            # do something
            logger.message('bar')  
        ```
        
        is equivalent to
        
        ```python
        logger.being('foo')
        # do something
        logger.message('bar')  
        logger.ended('foo')
        ```
    * The `message()` method can be called to write any information to the log.
    * Calling `begin()`, `ended()`, `subproc()` or `message()` on `smlb_out.log`
    means calling that function on `smlb_out.log.console`, `smlb_out.log.host` and
    `smlb_out.log.device` at the same time (see 
    [MNIST_tf_keras.py](sciml_bench/benchmarks/MNIST_tf_keras/MNIST_tf_keras.py)). 
    Nothing happens when these functions are called on an unactivated logger.
    * It is true that such logging will make the original code longer. However,
    *adding these lines should be a straightforward task because
    it involves only addition to the original code (instead of modification) 
    and has very simple structures.* 


* **Stamping events to system monitoring**

  System usage is sampled at a constant interval, as specified by `--monitor_interval` 
  in `sciml-bench run`. There is no explicit development is required to achieve the final system reports. To make these reports more readable, however, we provide a function for the contributors to stamp events in the time history of system monitoring:

  ```python
  smlb_out.system.stamp_event('any event')
  ```

  Unlike the logging, which is encouraged to be as detailed as possible throughout the runtime, 
  `system.stamp_event()` can be called only on the expensive subprocesses (such as training and validation) to identify the usage peaks.
  Taking `linreg_sklearn` for example, we can just stamp the `LinearRegression.fit()` method.

  The final code for `linreg_sklearn.py` is presented below (note that we are using `time.sleep()` to make the runtime and thus the reports a bit longer):

  ```python
  # libs from sciml_bench
  from sciml_bench.core.runtime import RuntimeIn, RuntimeOut
  
  # libs required by implementation
  import numpy as np
  from sklearn.linear_model import LinearRegression
  import time
  
  
  def sciml_bench_run(smlb_in: RuntimeIn, smlb_out: RuntimeOut):
      """ Main entry of `sciml_bench run` for a benchmark instance """
      # activate smlb_out
      smlb_out.activate(rank=0, local_rank=0)
  
      # begin top-level
      console = smlb_out.log.console
      console.begin('Running benchmark linreg_sklearn')
  
      # input
      with console.subproc('Creating some input data'):
          X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
          y = np.dot(X, np.array([1, 2])) + 3
  
      # get benchmark-specific arguments
      with console.subproc('Getting benchmark-specific arguments'):
          normalize = smlb_in.bench_args.try_get('normalize', default=False)
          console.message(f'normalize = {normalize}')
  
      # fit
      time.sleep(2)  # sleep, as if it is loading data
      console.begin('Training')
      with console.subproc('Fitting data'):
          smlb_out.system.stamp_event('LinearRegression.fit() begin')
          reg = LinearRegression(normalize=normalize).fit(X, y)
          time.sleep(10)  # sleep, as if it is doing fit()
          smlb_out.system.stamp_event('LinearRegression.fit() ended')
  
      # save coefficients and intercept
      with console.subproc('Saving coefficients and intercept'):
          np.savetxt(smlb_in.output_dir / 'coefficients.txt', reg.coef_)
          np.savetxt(smlb_in.output_dir / 'intercept.txt', [reg.intercept_])
      console.ended('Training')
      time.sleep(2)  # sleep, as if it is doing validation and inference
  
      # score
      with console.subproc('Computing score'):
          console.message(f'score = {reg.score(X, y)}')
  
      # predict
      with console.subproc('Making predictions'):
          reg.predict(np.array([[3, 5]]))
  
      # end top_level
      console.ended('Running benchmark linreg_sklearn')
  ```

  Finally, run `linreg_sklearn` with the following command and check the system reports in 
  `./linreg_sklearn/sciml_bench_run_sys/`.

  ```sh
  python -m sciml_bench.core.command run linreg_sklearn --output_dir=linreg_out --monitor_interval=0.1 -b normalize YES
  ```



## 3.5 Benchmark dependencies and documentation

The benchmark dependencies are handled by the function `install_benchmark_dependencies()` in
[benchmark.py](../sciml_bench/core/benchmark.py). For the `linreg_sklearn` example, we only need to insert two lines inside 
the loop under `# verify each benchmark` 
(duplicated dependencies are handled automatically): 

```python
# verify each benchmark
dependencies = set()
horovod_env = {}
for bench in benchmarks:
    if bench == 'MNIST_tf_keras':
        dependencies.add('tensorflow')
    elif bench == 'MNIST_torch':
        dependencies.add('torch')
        horovod_env['HOROVOD_WITH_PYTORCH'] = '1'
    elif bench == 'linreg_sklearn':  # added for linreg_sklearn
        dependencies.add('sklearn')  # added for linreg_sklearn
    ...
```



We will be in incorporating these dependencies along with other dependencies during the distribution stage.  

You are also required to provide a detailed scientific description of the  benchmark and datasets as part of your contribution. A summary of the benchmarks and datasets can be generated by running the [gen_bench_ds.py](./resources/gen_bench_ds.py) script within the `doc/resources/`  folder 

<div style="text-align: right">◼︎</div>

