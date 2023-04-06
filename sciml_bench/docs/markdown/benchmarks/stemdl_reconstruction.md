# stemdl_reconstruction Benchmark (to be updated)

The Scanning Transmission Electron Microscope Deep Learning (STEM-DL) benchmark was developed at Oakridge National Laboratory (ORNL). This benchmark trains a Neural Network model with diffraction data as input to reconstruct the material’s local electron density.The motivation for this benchmark is to explore the suitability of machine learning algorithms for the advanced analysis of CBED [1, 2, 3, 4, 5].

* Entity Type: Benchmark
* Main Domain: Material Sciences
* Sub Domain: Convergent Beam Electron Diffraction
* Task: Reconstruction of diffraction patterns
* Relevant Datasets: stemdl_reconst_ds1
* Implementation: Pytorch
* Authors: Junqi Yin, Sajal Dash, Aris Tsaris, Feiyi Wang, Arjun Shankar

The benchmark supports data parallel training and runs on multiple GPUs.
For training use this command: 

sciml-bench run stemdl_reconstruction --mode training -b epochs 1 -b batchsize 32 -b nodes 1 -b gpus 1

For inference use this command:
sciml-bench run stemdl_reconstruction --mode inference \
    --model ~/sciml_bench/outputs/stemdl_classification/stemdlModel.h5 \
    --dataset_dir ~/sciml_bench/datasets/stemdl_ds1 \
    -b epochs 1 -b batchsize 32 -b nodes 1 -b gpus 1 

## References:
[1] N Laanait, J Yin, “NAMSA”, 10.11578/dc.20201001.90, 2019.
[2] N Laanait, A Borisevich, J Yin, “Database of Convergent Beam Electron Diffraction Patterns for Machine Learning of the Structural Properties of Materials”, 10.13139/OLCF/1510313, 2019.
[3] N Laanait, J Yin, A Borisevich, “Towards a Universal Classifier for Crystallographic Space Groups”, SMC data challenges, 2020.
[4] Jin Pan, “Probability Flow for Classifying Crystallographic Space Groups”, SMC 2020.
[5] N Laanait, J Romero, J Yin, M Young, S Treichler, V Starchenko, A Borisevich, A Sergeev, M Matheson, “Exascale deep learning for scientific inverse problems”, arxiv:1909.11150, 2019.

<!--
The Scanning Transmission Electron Microscope Deep Learning (STEM-DL) benchmark was developed at Oakridge National Laboratory (ORNL). This benchmark trains a Neural Network model with diffraction data as input to reconstruct the material’s local electron density.The motivation for this benchmark is to explore the suitability of machine learning algorithms for the advanced analysis of CBED [1, 2, 3, 4, 5].

* Entity Type: Benchmark
* Main Domain: Material Sciences
* Sub Domain: Convergent Beam Electron Diffraction
* Task: Reconstruction of diffraction patterns
* Relevant Datasets: stemdl_reconst_ds1
* Implementation: Pytorch
* Authors: Junqi Yin, Sajal Dash, Aris Tsaris, Feiyi Wang, Arjun Shankar

The benchmark supports data parallel training and runs on multiple GPUs.
For training use this command: 

sciml-bench run stemdl_reconstruction --mode training -b epochs 1 -b batchsize 32 -b nodes 1 -b gpus 1

For inference use this command:
sciml-bench run stemdl_reconstruction --mode inference \
    --model ~/sciml_bench/outputs/stemdl_classification/stemdlModel.h5 \
    --dataset_dir ~/sciml_bench/datasets/stemdl_ds1 \
    -b epochs 1 -b batchsize 32 -b nodes 1 -b gpus 1 

## References:
[1] N Laanait, J Yin, “NAMSA”, 10.11578/dc.20201001.90, 2019.
[2] N Laanait, A Borisevich, J Yin, “Database of Convergent Beam Electron Diffraction Patterns for Machine Learning of the Structural Properties of Materials”, 10.13139/OLCF/1510313, 2019.
[3] N Laanait, J Yin, A Borisevich, “Towards a Universal Classifier for Crystallographic Space Groups”, SMC data challenges, 2020.
[4] Jin Pan, “Probability Flow for Classifying Crystallographic Space Groups”, SMC 2020.
[5] N Laanait, J Romero, J Yin, M Young, S Treichler, V Starchenko, A Borisevich, A Sergeev, M Matheson, “Exascale deep learning for scientific inverse problems”, arxiv:1909.11150, 2019.

-->