# hydronet_parallel

Parallel version of Hydronet

The modified version of Hydronet parallel can be cloned by this command:
git clone https://github.com/juripapay/hydronet_parallel
In the modified version the data loading was changed. 

The first DDP version of the code was developed by Firoz, Jesun S <jesun.firoz@pnnl.gov>.
The code can be downloaded from the schnet_ddp branch:
git clone --branch schnet_ddp https://github.com/jenna1701/hydronet/

For running the parallel version of hydroned we need to create a conda environment with 
dependencies:
conda create --name hydronet2 --file hydronet_dependencies.txt

Runing the code:
-------------
conda activate hydronet2

torchrun --standalone --nnodes=1  --nproc_per_node=2  train_direct_ddp.py --savedir './test_train_ddp1' --args 'train_args.json'
