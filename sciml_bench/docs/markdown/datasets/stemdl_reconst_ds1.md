# stemdl_reconst_ds1 Dataset (to update)

The stemdl_reconst_ds1 dataset is coming from Oakridge National Laboratory (ORNL). The data is in LMDB format. More details at https://code.ornl.gov/jqyin/stemdl-benchmark/-/blob/master/reconstruction/stemdl/inputs.py#L453.
Each batch_train directory corresponds to an input data for one GPU. So for 32 GPUs, we need 32 batch_train directory from 0 to 31. 

* Data Source: Generated diffraction images
* Domain: Sciences
* Sub-Domain: Material science
* Data Type: Images 
* Enclosing Benchmark: stemdl_reconstruction
* Data Size: 100GB * number_of_GPUs

<!--
The stemdl_reconst_ds1 dataset is coming from Oakridge National Laboratory (ORNL). The data is in LMDB format. More details at https://code.ornl.gov/jqyin/stemdl-benchmark/-/blob/master/reconstruction/stemdl/inputs.py#L453.
Each batch_train directory corresponds to an input data for one GPU. So for 32 GPUs, we need 32 batch_train directory from 0 to 31. 

* Data Source: Generated diffraction images
* Domain: Sciences
* Sub-Domain: Material science
* Data Type: Images 
* Enclosing Benchmark: stemdl_reconstruction
* Data Size: 100GB * number_of_GPUs
-->