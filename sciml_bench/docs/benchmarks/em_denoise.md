# em-denoise Benchmark

Increased frame collection rates on modern electron microscopes allows for the observation of dynamic processes, such as defect migration and surface reconstruction. To capture these dynamic processes, images are often collected at a high frequency, resulting in huge volumes of data.In almost all instances where micrographs are analysed, it is desirable to have techniques to improve signal to noise ratios of the images. Effective denoising can facilitate low-dose experiments, with image quality comparable to high-dose experiments. Likewise greater time resolution can be achieved with the aid of effective image denoising procedures. 

This benchmark includes seven baseline models - Class-aware Fully Convolutional Networks, De-noising CNN, FFD-Net, U-Net, Deep Encoder Decoder with Skip Connections, Multiscale CNN, and Mixed Scale Dense Networks. 

* Main Domain: Material Sciences
* Sub Domain: Condensed Matter Physics
* Task: Estimation
* Relevant Datasets: em_graphene_sim
* Implementation: PyTorch
* Support for Distributed Training: No
* Device Support: CPU / GPU
* Authors: Keith Butler, Patrick Austin, Jeyan Thiyagalingam, and Tony Hey 

The benchmark, when run, by default, utilises the GPU if one available.  The benchmark does not support distributed training, and such, will not make use of more than one GPU, even if available. The benchmark relies on the following default parameters: 

* Batch Size: `batch_size` = 128
* Number of Epochs: `epochs` = 2
* Learning Rate: `lr` = 0.01

These can be tuned further by supplying the values of when running the benchmark in training mode using the `--benchmark_specific` or `-b` option.   

The model file for the *em_denoise* is stored as a single file in HDF format under the name of `em_denoise_model.h5` in the outputs folder. 




<!--
Increased frame collection rates on modern electron microscopes allows for the observation of dynamic processes, such as defect migration and surface reconstruction. To capture these dynamic processes, images are often collected at a high frequency, resulting in huge volumes of data.In almost all instances where micrographs are analysed, it is desirable to have techniques to improve signal to noise ratios of the images. Effective denoising can facilitate low-dose experiments, with image quality comparable to high-dose experiments. Likewise greater time resolution can be achieved with the aid of effective image denoising procedures. 

This benchmark includes seven baseline models - Class-aware Fully Convolutional Networks, De-noising CNN, FFD-Net, U-Net, Deep Encoder Decoder with Skip Connections, Multiscale CNN, and Mixed Scale Dense Networks. 

* Entity Type: Benchmark
* Main Domain: Material Sciences
* Sub Domain: Condensed Matter Physics
* Task: Estimation 
* Relevant Datasets: em_graphene_sim
* Implementation: PyTorch
* Authors: Keith Butler, Patrick Austin, 
           Jeyan Thiyagalingam, and Tony Hey 


-->

