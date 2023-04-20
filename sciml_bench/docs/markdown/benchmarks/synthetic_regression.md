# synthetic_regression Benchmark

This benchmark is a synthetic regression model designed to exercise the GPU with a heavy workload. The benchmark genearates random regression problems and attempts to fit them using a simple feedforward MLP. This benchmark supports distributed multi-GPU training using `torchrun`.

* Main Domain: Synthetic
* Sub Domain: N/a
* Task: Regression
* Relevant Datasets: N/a
* Implementation: PyTorch Lightning
* Support for Distributed Training: Yes
* Device Support: GPU
* Authors: Juri Papay, Samuel Jackson, and Jeyan Thiyagalingam

The benchmark, when run, by default, utilises the GPU if one is available.  The benchmark does not support distributed training, and such, will not make use of more than one GPU, even if available. The benchmark relies on the following default parameters: 

* Input size: `input_size` = 784
* Batch size: `batch_size` = 128
* Number of samples: `num_samples` = 1024000
* Hidden layer size: `hidden_size` = 3000
* Epochs: `epochs` = 1

These can be tuned further by supplying the values of when running the benchmark in training mode using the `--benchmark_specific` or `-b` option.   

<!--
This benchmark is a synthetic regression model designed to exercise the GPU with a heavy workload. The benchmark genearates random regression problems and attempts to fit them using a simple feedforward MLP. This benchmark supports distributed multi-GPU training using `torchrun`.

* Main Domain: Synthetic
* Sub Domain: N/a
* Task: Regression
* Relevant Datasets: N/a
* Implementation: PyTorch Lightning
* Support for Distributed Training: Yes
* Device Support: GPU
* Authors: Juri Papay, Samuel Jackson, and Jeyan Thiyagalingam
-->