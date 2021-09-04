# optical_damage Benchmark

(udate this section)
Estimation of sea surface temperature (SST) from space-borne sensors, such as satellites, is crucial for a number of applications in environmental sciences. One of the aspects that underpins the derivation of SST is cloud screening, which is a step that marks each and every pixel of thousands of satellite imageries as containing cloud or clear sky, historically performed using either thresholding or Bayesian methods. This benchmark focuses on using a machine learning-based model for masking clouds, in the Sentinel-3 satellite, which carries the Sea and Land Surface Temperature Radiometer (SLSTR) instrument. More specifically, the benchmark operates on multi-spectral image data. The baseline implementation is a variation of the U-Net deep neural network. 


* Main Domain: Environmental Sciences
* Sub Domain: Atmospheric Physics / Remote Sensing
* Task:	Image classification (at pixel level)
* Relevant Datasets: cloud_slstr_ds1, cloud_slstr_ds2
* Implementation: TensorFlow + Keras + Horovod
* Support for Distributed Training: Yes
* Device Support: CPU / GPUs / Clusters
* Authors: Samuel Jackson, Caroline Cox, Jeyan Thiyagalingam, and Tony Hey 


The benchmark, when run, by default, utilises the GPU if one available.  The benchmark does not support distributed training, and such, will not make use of more than one GPU, even if available. The benchmark relies on the following default parameters: 

* Batch size: `batch_size` = 32
* Number of epochs: `epochs` = 30
* Learning rate: `lr` = 0.001
* Clipping offset: `clip_offset` = 15
* Train/Test split ratio `train_split` = 0.8
* Cropping size (to remove potential NaN values) `crop_size` = 80
* Use TensorFlow dataset caching during training `no_cache` = False
* Value for the weighted binary cross entropy `wbce`  = 0.5

These can be tuned further by supplying the values of when running the benchmark in training mode using the `--benchmark_specific` or `-b` option.  


<!--
Estimation of sea surface temperature (SST) from space-borne sensors, such as satellites, is crucial for a number of applications in environmental sciences. One of the aspects that underpins the derivation of SST is cloud screening, which is a step that marks each and every pixel of thousands of satellite imageries as containing cloud or clear sky, historically performed using either thresholding or Bayesian methods. This benchmark focuses on using a machine learning-based model for masking clouds, in the Sentinel-3 satellite, which carries the Sea and Land Surface Temperature Radiometer (SLSTR) instrument. More specifically, the benchmark operates on multi-spectral image data. The baseline implementation is a variation of the U-Net deep neural network. 

* Entity Type: Benchmark
* Main Domain: Environmental Sciences
* Sub Domain: Atmospheric Physics / Remote Sensing
* Learning Task: Image classification (at pixel level)
* Relevant Datasets: cloud_slstr_ds1, cloud_slstr_ds2
* Implementation: TensorFlow + Keras + Horovod
* Support for Distributed Training: Yes
* Device Support: CPU / GPUs / Clusters
* Authors: Samuel Jackson, Caroline Cox, 
           Jeyan Thiyagalingam, and Tony Hey 
 
  Check the full documentation for more details. 
-->