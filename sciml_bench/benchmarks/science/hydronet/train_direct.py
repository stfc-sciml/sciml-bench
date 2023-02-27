# NEED TO: conda install tensorboard
'''
1) Activate conda env
conda activate hydronet2
2) Run
python train_direct.py --savedir './trained_model' --args 'train_args.json'

'''

import os, sys, time, yaml, math, json, csv, argparse, shutil, logging
from pathlib import Path
import torch
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.nn import DataParallel
from tqdm import tqdm
#from utils import data, models, train, eval, split, hooks
import datetime

# integration libs
from sciml_bench.core.runtime import RuntimeIn, RuntimeOut

import sciml_bench.benchmarks.science.hydronet.utils.data
import sciml_bench.benchmarks.science.hydronet.utils.models
import sciml_bench.benchmarks.science.hydronet.utils.train
import sciml_bench.benchmarks.science.hydronet.utils.eval
import sciml_bench.benchmarks.science.hydronet.utils.split
import sciml_bench.benchmarks.science.hydronet.utils.hooks


# Timestamp 
start_time = datetime.datetime.now()
print(f'Start: {start_time}')
sys.exit()

# logging all to stdout
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

# import path arguments
parser = argparse.ArgumentParser()
parser.add_argument('--savedir', type=str, required=True, help='Directory to save training results')
parser.add_argument('--args', type=str, required=True, help='Path to training arguments')
args = parser.parse_args()

######## SET UP ########
# create directory to store training results
if not os.path.isdir(args.savedir):
    os.mkdir(args.savedir)
    os.mkdir(os.path.join(args.savedir,'tensorboard')) 
else:
    logging.warning(f'{args.savedir} is already a directory, either delete or choose new SAVEDIR')
    sys.exit()
    
# set up tensorboard logger
writer = SummaryWriter(log_dir=os.path.join(args.savedir,'tensorboard'))

# copy args file to training folder
shutil.copy(args.args, os.path.join(args.savedir, 'args.json'))

# read in args
savedir = args.savedir
with open(args.args) as f:
    args_dict = json.load(f)
    args = argparse.Namespace(**args_dict)
args.savedir = savedir

# check for GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('torch available',torch.cuda.is_available())
logging.info(f'model will be trained on {device}')

# increase batch size based on the number of GPUs
if device == 'cuda' and args.parallel:
    n_gpus = torch.cuda.device_count()
    args.batch_size = int(n_gpus * args.batch_size)
    logging.info(f'... {n_gpus} found, multipying batch size accordingly (batch size now {args.batch_size})')

######## LOAD DATA ########
# get initial train, val, examine splits for dataset(s)
if args.create_splits:
    logging.info('creating new split(s)')
    # create split(s)
    if not isinstance(args.datasets, list):
        args.datasets = [args.datasets]
    for dataset in args.datasets:
        split.create_init_split(args, dataset) 
else:
    # copy initial split(s) to savedir
    logging.info(f'starting from splits in {args.splitdir}')
    if isinstance(args.datasets, list):
        for dataset in args.datasets:
            shutil.copy(os.path.join(args.splitdir, f'split_00_{dataset}.npz'), 
                        os.path.join(args.savedir, f'split_00_{dataset}.npz'))
    else: 
        shutil.copy(os.path.join(args.splitdir, f'split_00_{args.datasets}.npz'), 
                    os.path.join(args.savedir, f'split_00_{args.datasets}.npz'))

# load datasets/dataloadersls

train_loader, val_loader = data.init_dataloader(args)
    
######## LOAD MODEL ########

# load model
net = models.load_model(args, device=device)
if device == 'cuda' and args.parallel: 
    net = DataParallel(net)
logging.info(f'model loaded from {args.start_model}')

#initialize optimizer and LR scheduler
optimizer = torch.optim.Adam(net.parameters(), lr=args.start_lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.8, min_lr=0.000001)

# implement early stopping
early_stopping = hooks.EarlyStopping(patience=500, verbose=True, 
                                     path = os.path.join(args.savedir, 'best_model.pt'),
                                     trace_func=logging.info)

total_epochs = 0
for _ in tqdm(range(args.n_epochs)):

    if args.train_forces:
        train_loss, e_loss, f_loss = train.train_energy_forces(args, net, train_loader, optimizer, device)
        val_loss = train.get_pred_loss(args, net, val_loader, optimizer, device)
    else:
        train_loss = train.train_energy_only(args, net, train_loader, optimizer, device)
        val_loss = train.get_pred_eloss(args, net, val_loader, optimizer, device)

    scheduler.step(val_loss)
    
    # log training info
    writer.add_scalars('epoch_loss', {'train':train_loss,'val':val_loss}, total_epochs)
    if args.train_forces:
        writer.add_scalars('train_loss', {'energy':e_loss,'forces':f_loss}, total_epochs)
    writer.add_scalar(f'learning_rate', optimizer.param_groups[0]["lr"], total_epochs)
    
    # check for stopping point
    early_stopping(val_loss, net)
    if early_stopping.early_stop:
        break

    total_epochs+=1

# close tensorboard logger
writer.close()

# Program end timestamp
end_time = datetime.datetime.now() 
run_time = end_time - start_time
print(f'End Time: {end_time}')
print(f'run_time: {run_time}')

