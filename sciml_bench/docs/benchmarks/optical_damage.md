# optical_damage Benchmark

The purpose of this benchmark is to detect damaged and degrading laser optics from diagnostic images for the optics beam path. Damaged laser optics are a significant problem in Central Laser Facility (CLF) beamlines. Damaged optics are not always spotted automatically by users, causing a loss of productive time and a potentially a corruption of scientific results. Depending on the type and extent of the damage, optics could also be potentially salvaged if damage can be detected early enough. The benchmark uses an autoencoder which is trained on a large number of images of undamaged optics and the faulty equipment is detected as an anomaly. The solution is based on unsupervised learning for several reasons. Firstly, images of damaged optics are uncommon and exhibit a lot of variation. Secondly, it can be difficult to design a network that can spot all types of damage when trained in a supervised fashion. Therefore the problem is framed as outlier detection, the network learns what an undamaged image looks like then it can spot the outliers (damaged images using the reconstruction loss). In autoencoder networks, the original image is encoded to produce a compressed data representation then this representation is decoded to reconstruct the original image or at least a close resemblance. An anomaly can be detected by determining how well the model can reconstruct the input data. Since the model was trained on undamaged imaged we would expect that a high reconstruction error for images of damaged optics. The autoencoder consists of 160 million trainable parameters and it occupies 1.8GBytes of RAM.
 
* Entity Type: Benchmark
* Main Domain: Instrumentation
* Sub Domain: Laser equipment
* Task: Detection of damaged optical equipment 
* Relevant Datasets: optical_damage_ds1
* Implementation: TensorFlow Keras
* Authors: Samuel Jackson, Nicholas Bourgeois, Stephen Dann

The benchmark, when run, by default, utilises the GPU if one available.  The benchmark does not support distributed training, and such, will not make use of more than one GPU, even if available. 

For training use this command:
sciml-bench run --mode training optical_damage

For inference use this command:
sciml-bench run --mode inference --model ~/sciml_bench/models/optical_damage/opticsModel.h5 --dataset_dir ~/sciml_bench/datasets/optical_damage_ds1 optical_damage

The benchmark uses on the following default parameters (update this section): 

* Batch size: `batch_size` = 32
* Number of epochs: `epochs` = 1
* Learning rate: `lr` = 0.01  
