'''
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

For detailed log use this command:
TORCH_DISTRIBUTED_DEBUG=INFO NCCL_DEBUG=INFO TORCH_CPP_LOG_LEVEL=INFO python -m torch.distributed.run --nnodes=1  --nproc_per_node=2  train_direct_ddp.py --savedir './test_train_ddp2' --args 'train_args_min.json'

'''

import os, sys
import torch
import shutil
import logging
import json 
import csv
import argparse
import time

from torch.utils.tensorboard import SummaryWriter

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from tqdm import tqdm
from utils import data_ddp, models, train, train_ddp, models_ddp, eval, split, hooks

def main(args):
    os.environ["TORCH_CPP_LOG_LEVEL"]="INFO"
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"  # set to DETAIL for runtime logging.

    dist.init_process_group(backend="nccl", init_method='env://')
    rank = int(os.environ["LOCAL_RANK"])  # dist.get_rank()
    print(f"Start running SchNet on rank {rank}.")
    CUDA_VISIBLE_DEVICES = int(os.environ["LOCAL_RANK"])                              

    n_gpus = dist.get_world_size()
    device_id = rank

    torch.cuda.set_device(device_id)

    print(
        f"[{os.getpid()}]: world_size = {dist.get_world_size()}, "
        + f"rank = {dist.get_rank()}, backend={dist.get_backend()} \n", end=''
    )
    print(f'num_gpus: {n_gpus}')
    print("Running the DDP model")

    # increase batch size based on the number of GPUs
    #args.batch_size = int(n_gpus * args.batch_size)
    # For larger batches increase the learning rate.
    local_batch_size = 128
    global_batch_size = n_gpus * local_batch_size
    # Linear LR scaling
    learning_rate = args.start_lr * n_gpus

    logging.info(f'... {n_gpus} found, multipying batch size accordingly (batch size now {global_batch_size})')
    
    ######## SET UP ########
    # create directory to store training results
    if device_id == 0:
        if not os.path.isdir(args.savedir):
            os.mkdir(args.savedir)
            os.mkdir(os.path.join(args.savedir,'tensorboard')) 
        else:
            logging.warning(f'{args.savedir} is already a directory, either delete or choose new SAVEDIR')
            # sys.exit()
    
        # copy args file to training folder
        shutil.copy(args.args, os.path.join(args.savedir, 'args.json'))

    dist.barrier()
    #torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
    
    # read in args
    savedir = args.savedir
    with open(args.args) as f:
        args_dict = json.load(f)
        args = argparse.Namespace(**args_dict)
    args.savedir = savedir
    
    # check for GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device ", device)
    logging.info(f'model will be trained on {device}')

    torch.manual_seed(12345)
    net = models_ddp.load_model_ddp(args, device_id, device)

    # criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.8, min_lr=0.000001)
    
    # load datasets/dataloaders
    train_loader, val_loader, train_sampler = data_ddp.init_dataloader(args, local_batch_size)
   
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
        
    start.record()
   
    for epoch in range(args.n_epochs):
        net.train()
        # let all processes sync up before starting with a new epoch of training
        dist.barrier()
        #torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
       
        # DistributedSampler deterministically shuffle data
        # by seting random seed be current number epoch
        # so if do not call set_epoch when start of one epoch
        # the order of shuffled data will be always same
        train_sampler.set_epoch(epoch)
        epoch_start_time = time.time()
        
        print("type(train_loader): ", type(train_loader))
        train_loss = train_ddp.train_energy_only_ddp(args, device_id, 
                                                     net, train_loader, optimizer, 
                                                     device)

        dist.barrier()
        
        val_loss = train_ddp.get_pred_eloss_ddp(args, device_id, net, 
                                            val_loader, optimizer, device)
        print("-" * 89, flush=True)
        print("| end of epoch {:3d} | time: {:5.2f}s |  valid loss {:5.4f} |".format(
                epoch, (time.time() - epoch_start_time), val_loss), flush=True)
        scheduler.step(val_loss) 
        dist.barrier()

    end.record()
    dist.destroy_process_group()
    # Waits for everything to finish running
    torch.cuda.synchronize()
        
    print(f'Elapsed time in miliseconds {start.elapsed_time(end)}')  # milliseconds

if __name__ == '__main__':
    # import path arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--savedir', type=str, required=True, help='Directory to save training results')
    parser.add_argument('--args', type=str, required=True, help='Path to training arguments')

    # # This is passed in via launch.py
    # parser.add_argument("--local_rank", type=int, default=0)
    # # This needs to be explicitly passed in
    # parser.add_argument("--local_world_size", type=int, default=1)

    args = parser.parse_args()
    main(args)

    # train_dataloader, val_dataloader = load_data(args)

    # world_size = torch.cuda.device_count()
    # print('Let\'s use', world_size, 'GPUs using DistributedDataParallel!')
    # mp.spawn(run, args=args, nprocs=world_size, join=True)
