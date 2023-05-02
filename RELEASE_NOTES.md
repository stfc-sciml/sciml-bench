# RELEASE NOTES

# Version 1.1.0.b


## Documentation

1. Added full documentation of entities (such as benchmarks, datasets and models). The new release now includes a full-fledged HTML-based documentation, powered by MKDocs. 

1. A detailed documentation has been now added around all benchmarks, datasets,  installation aspects.

1. A detailed FAQ and Tutorial have also been added. 

1. Updated CREDITS file. 


## Structure and Organisation 

1. Reorganisation of the file structure.

1. Simplified the benchmark, datasets and framework options using a single Configuration file. 

1. Removed the use of files for retaining logo and messages. They are now incorporated as part of the files. This results in improved response to a number of commands, such as `about`. 

1. The new version has an improved API to support / separate training and inference aspects of benchmarking. 

1. Notion of benchmark groups has been introduced.  This supports better grouping of benchmarks into sub-folders. 


## Commands and Options 

1. Additional commands: `info`, `version`.

1. Additional options for commands:
    * `--deps` option for the `list` command for obtaining dependency information on benchmarks,
    * `--mode`, `--model`, `--dataset_dir` options for the `run` command, and
    * `--mode` option for the `download` command.

1. Removal of redundant / nonsensical features, commands and options from 1.0.0
    * Removal of `--brief` option for the `list` command. 

1. Very streamlined and cleaner interface for download, and install commands. All download and installation commands result in summary outputs and detailed outputs are available as log files. This means, less clutter to the terminal window. 

1. Numerous bug fixes, enhancements and simplification of options and commands. 

1. Added `--deps` option to the list command to list dependencies of benchmarks 

1. Added download check for `--verify` option.



## Benchmarks 

1. Separation of example benchmarks/datasets from mainstream benchmarks/datasets.

1. Converted the *em_denoise* benchmark to PyTorch for better support. 

1. The *mnist* example is now available in PyTorch, TensorFlow+Keras, and Apache MXNet. 

1. Simplified all benchmark implementations. New benchmark implementations are well structured with less clutter.

1. All benchmarks now include inference options. 


## Datasets 

1. Datasets are now available in two locations: STFC (UK) and Open Storage Network (San Diego Supercomputing Centre). 

◼︎
