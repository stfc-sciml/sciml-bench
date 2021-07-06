
# Using the Suite and Benchmarks
<br>
<br>

## 1 Verifying Benchmarks

Before running a benchmark,  any issues with the relevant datasets and modules can be verified via the verify flag. 

```sh
sciml-bench list --verify
```

The results from the verify flag indicates whether the relevant dataset(s) has/have been downloaded,  and whether the dependencies of the benchmark are fully satisfied. If the verify command outputs  *Not Runnable* for any benchmark, the relevant datasets dependencies have to be downloaded /  installed. The *Not Runnable*  output can also be the result of benchmark codes not being able to import the dependencies.

<br> 


## 2 Running Benchmarks Using the Framework


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


<br> 

## 3 Running Benchmarks Using Container Images

If the benchmarks were to be run on a production cluster, ideally they may have to be run using one of the container technologies, such as Singularity of Docker. Although both methods need elevated privileges (equivalent to root access) for building containers, Singularity does not demand elevated privileges for running the containers. Please see Section [Building Containers]() about building containers. 

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

<br> 

##  4 Outputs of the Runs 

Outputs of the benchmark runs are automatically logged into a sub-directory inside the main output directory, which is specified in the Configuration file. This is, by default, set to `~/sciml_bench/outputs/` but can be overridden.  The sub-directory name is same as the benchmark name, followed by date of run and type of run (training or inference). All the arguments used for a run are usually saved to the `arguments_used.yml` file in this  directory, subject to individual benchmarks deciding to do this. The sub-directory will also contain all log files capturing the outputs of the run. 
