# dms_structure Benchmark

Diffuse Multiple Scattering (DMS) is a phenomenon that has been observed in X-ray patterns for many years, but has only become accessible as a useful tool for analysis with the advent of modern X-ray sources and sensitive detectors in the past decade. The method is very promising, allowing for investigation of multi-phase materials from a single measurement – something not possible with standard X-ray experiments. However, analysis currently relies on extremely laborious searching of patterns to identify important motifs (triple intersections) that allow for inference of information. This task can only be performed by expert beam scientists and severely limits the application of this promising technique. This benchmark aims to measure the possibility of automating this. There are two benchmark cases here: (i) a simple binary classification to distinguish between two possible phases, and (ii) A multi phase classification problem where we classify between six different phases or (possibly) combinations of two of the phases.

* Main Domain: Material Sciences
* Sub Domain: Condensed Matter Physics
* Task:	Image classification 
* Relevant Datasets: dms_sim
* Implementation: PyTorch
* Support for Distributed Training: No
* Device Support: CPU / GPU
* Authors: Keith Butler, Patrick Austin, Jeyan Thiyagalingam, and Tony Hey 

The benchmark, when run, by default, utilises the GPU if one available.  The benchmark does not support distributed training, and such, will not make use of more than one GPU, even if available. The benchmark relies on the following default parameters: 

* Batch Size: `batch_size` = 32
* Number of Epochs: `epochs` = 30
* Learning Rate: `lr` = 0.0001
* Early termination threshold: `patience` = 10

These can be tuned further by supplying the values of when running the benchmark in training mode using the `--benchmark_specific` or `-b` option.  

The model file for the *dms_structure* is stored as a single file in HDF format under the name of `dms_structure_model.h5` in the outputs folder. 


<!--
Diffuse Multiple Scattering (DMS) is a phenomenon that has been observed in X-ray patterns for many years, but has only become accessible as a useful tool for analysis with the advent of modern X-ray sources and sensitive detectors in the past decade. The method is very promising, allowing for investigation of multi-phase materials from a single measurement – something not possible with standard X-ray experiments. However, analysis currently relies on extremely laborious searching of patterns to identify important motifs (triple intersections) that allow for inference of information. This task can only be performed by expert beam scientists and severely limits the application of this promising technique. This benchmark aims to measure the possibility of automating this. There are two benchmark cases here: (i) a simple binary classification to distinguish between two possible phases, and (ii) A multi phase classification problem where we classify between six different phases or (possibly) combinations of two of the phases.

* Entity Type: Benchmark
* Main Domain: Material Sciences
* Sub Domain: Condensed Matter Physics
* Task:	Image classification 
* Relevant Datasets: dms_sim
* Implementation: PyTorch
* Support for Distributed Training: No
* Device Support: CPU / GPU
* Authors: Keith Butler, Patrick Austin, Jeyan Thiyagalingam, and Tony Hey 
* Authors: Keith Butler, Gareth Nisbet, Steve Collins
           Jeyan Thiyagalingam, and Tony Hey
-->