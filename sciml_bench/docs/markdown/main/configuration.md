# Configuration Options 

SciMLBench relies on a single configuration file, `config.yml`, inside the `$SCIMLBENCHROOT/etc/configs/` folder. The configuration file has the following sections:

1. Data Mirrors, 
2. Download Commands, 
3. Directories, 
4. Datasets, and 
5. Benchmarks.

Each of these sections contains subsections or (key:value) pairs.  These are discussed in the following subsections.

## Data Mirrors

SciMLBench datasets are mirrored at various locations, and this section includes the URLs for those mirrors. Usually, these are not supposed to be modified. However, if you are developing a benchmark, you can add your own servers. The framework decides which mirrors based on the bandwidth information, but this can be overridden. 

## Download Commands

A number of different types of download commands can be defined, and then can be used in latter sections / subsections. Some examples of download commands are: 

```yaml
    download_command1: "aws s3 --no-sign-request --endpoint-url $SERVER sync $DATASET_URI $DATASET_DIR"
    download_command2: "wget -c https://www.dropbox.com/s/6xhviply27tw3yu/mnist.tar.bz2 -O -
            | tar -jx -C $DATASET_DIR"
```

where variables like `$SERVER`, `$DATASET_URI` and `$DATASET_DIR` refers to one of the data mirrors (defined above), location of the dataset in the mirror, and local directory where the dataset should be saved (see below). 


## Directories 

This section specifies where the downloaded datasets, outputs of the benchmark runs and downloaded pre-trained models, to be saved locally on the system, respectively.  An example entry would look like the following:

```yaml
directories:
    dataset_root_dir: ~/sciml_bench/datasets
    output_root_dir: ~/sciml_bench/outputs
    models_dir: ~/sciml_bench/models
```
In the case of containers, these directories can be specified as part of the run. 

## Datasets 

Datasets are included in the main section `datasets`. Each benchmark can have following sub-sections: datasets, *is_example*, *download_command*, and *end_point*.

### `is_example`
Possible Values are `True` or `False`. This flag indicates whether this dataset is an example / demo dataset. Functionally, this does not make any  difference, but can be listed separately using the `list` command. If ignored, it is assumed as `False`.

### `end_point`

The datasets are usually supplied from one of the STFC servers or mirrors. This should not be modified, unless you are developing your own benchmark and testing this. This end point is used as part of the predefined download command. If you are using your own download command, you may ignore this variable. 

### `download_command`

This can be one of the predefined variables from the `download_commands` section, or can be a raw string. This is used to download the actual dataset. An example of using a predefined download command is:


```sh
datasets: 
    mnist:
        end_point: 'sciml-datasets/ts'
        download_command: download_command1
```

In this case, the dataset is assumed to be a non-example (`is_example=False`) and will be downloaded using the predefined command `download_command1` defined in the `download_commands` section. An alternative example, where a direct command being used is:

```sh
datasets: 
    mnist:
        end_point: 'sciml-datasets/ts'
        download_command:  "wget -c https://www.server.com/mnist.tar.bz2 -O - | tar -jx -C DATASET_DIR"
```

The variable `DATASET_DIR` is used to ensure that the downloaded files are stored inside `dataset_dir` directory specified in the `directories` section.


## Benchmarks 

Benchmarks are included in the main section `benchmarks`. Each benchmark can have following sub-sections: datasets, *is_example*, *dependencies*, and *types*.

### `datasets`
These are dataset(s) used by the current benchmark. A benchmark can rely on one or more datasets, and can be specified using comma separated values. However, a corresponding entry must exist in the `datasets` section.

### `is_example`
Possible Values are `True` or `False`. This flag indicates whether this benchmark is an example / demo benchmark. Functionally, this does not make any  difference, but can be listed separately using the `list` command and can be used as templates for building complex benchmarks. 

### `dependencies`
These are actual library dependencies that the current benchmark relies on specified inside a string in a comma separated form. An example entrY would be: `torch, tensorflow, scikit-learn,horovod.torch`. For the framework to succeed, it must be possible to install each of these dependencies using the `pip install` command if the user opts to do so. Not all dependencies are easy to manage. For instance, horovod-based dependencies are known to be very difficult, and may warrant separate manual installation at the system level. These dependencies can also be platform specific. Regardless whether they have been pre-installed or already available on the system, these dependencies must be specified for the framework to function correctly.


### `types`
Specifies the type of benchmark, and permissible values are '*training*', '*inference*' or '*training,inference*' (or *'inference,training'*). This provides a way for the framework to know whether a particular benchmark supports training or inference or both. If ignored, the framework would assume that the benchmark supports both training and inference. If the benchmark actually does not support any of these, it will fail at runtime. An example section of the Configuration file is:

```yaml
benchmarks:
    dms_structure:
        datasets: dms_sim
        dependencies: 'torch'
        types: 'training, inference'
```




