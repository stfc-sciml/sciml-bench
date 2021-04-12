# Registered Datasets and Benchmarks

## Benchmarks

<TABLE style="width:100%>"
<TR>
<TH>Benchmark</TH>
<TH>Dataset </TH>
<TH>Title </TH>
<TH>Info </TH>
<TH>Dependencies </TH>
</TR>
<TR>
<TD>MNIST_tf_keras</TD>
<TD>MNIST</TD>
<TD>Classifying MNIST with CNN using Tensorflow Keras.</TD>
<TD>Demonstrates how to build a benchmark into SciML-Bench.</TD>
<TD>tensorflow</TD>
</TR>
<TR>
<TD>MNIST_torch</TD>
<TD>MNIST</TD>
<TD>Classifying MNIST with CNN using Pytorch and Horovod for distributed learning.</TD>
<TD>Demonstrates how to build a benchmark into SciML-Bench.</TD>
<TD>torch, horovod (with HOROVOD_WITH_PYTORCH=1)</TD>
</TR>
<TR>
<TD>em_denoise</TD>
<TD>em_graphene_sim</TD>
<TD>Denoising electron microscopy (EM) images of graphene using an autoencoder.</TD>
<TD>Here the datasets are simulated datasets.</TD>
<TD>mxnet</TD>
</TR>
<TR>
<TD>dms_structure</TD>
<TD>dms_sim</TD>
<TD>Classifying crystal structures based on the DMS pattern.</TD>
<TD>Diffuse multiple scattering patterns simulated for Tetragonal and Rhombohedral crystal strcutures mimics data collecet at Diamond Light Source.</TD>
<TD>torch</TD>
</TR>
<TR>
<TD>slstr_cloud</TD>
<TD>slstr_cloud_ds1</TD>
<TD>Cloud segmentation in Sentinel-3 SLSTR images</TD>
<TD>Classifying pixels as either cloudy or clear using images from the SLSTR instrument onboard Sentinel-3 using a U-Net style architecture.</TD>
<TD>tensorflow, horovod (with HOROVOD_WITH_TENSORFLOW=1), scikit-learn</TD>
</TR>
</TABLE>


Please see [CREDITS](./credits.md) for further information.




## Datasets

<TABLE style="width:100%>"
<TR>
<TH>Dataset</TH>
<TH>Size (approx) </TH>
<TH>Title </TH>
<TH>Info </TH>
<TH>Data server </TH>
</TR>
<TR>
<TD>MNIST</TD>
<TD>12 MB</TD>
<TD>The MNIST database of handwritten digits.</TD>
<TD>Demonstrates how to add a dataset to SciML-Bench.</TD>
<TD>By contributors</TD>
</TR>
<TR>
<TD>em_graphene_sim</TD>
<TD>28 GB</TD>
<TD>Simulated electron microscopy (EM) images of graphene.</TD>
<TD>Each image has a clean and a noisy version.</TD>
<TD>By contributors</TD>
</TR>
<TR>
<TD>dms_sim</TD>
<TD>7 GB</TD>
<TD>Simulated diffuse multiple scattering (DMS) patterns.</TD>
<TD>The patterns are labelled by the azimuthal angles.</TD>
<TD>By contributors</TD>
</TR>
<TR>
<TD>slstr_cloud_ds1</TD>
<TD>180 GB</TD>
<TD>Sentinel-3 SLSTR satellite image data.</TD>
<TD>The ground truth of a pixel as either cloudy or clear is provided.</TD>
<TD>By contributors</TD>
</TR>
</TABLE>


Please see [CREDITS](./credits.md) for further information.
<div style="text-align: right">◼︎</div>

