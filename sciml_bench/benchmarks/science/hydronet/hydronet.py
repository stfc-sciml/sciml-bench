import os
import time
import torch
import yaml
import numpy as np
import pandas as pd
import os.path as op
from sciml_bench.core.runtime import RuntimeIn, RuntimeOut

import torch.distributed as dist
from torch_geometric.loader import DataLoader
from sciml_bench.benchmarks.science.hydronet.utils import data_ddp, train_ddp, models_ddp
import sciml_bench.benchmarks.science.hydronet.utils.infer


def sciml_bench_training(params_in: RuntimeIn, params_out: RuntimeOut):
    """
    Main entry of `sciml_bench run` for a benchmark instance in the training mode.
    """

    default_args = {
        "parallel": True,
        "create_splits": False,
        "n_train": 107108,
        "n_val": 13388,
        "splitdir": "~/sciml_bench/datasets/hydronet_ds1/QM9/",
        "datadir": "sciml_bench/datasets/hydronet_ds1/",
        "inputFile" : "qm9.hdf5",
        "savedir": "trainingResults",
        "train_forces": False,
        "energy_coeff": 1,
        "n_epochs": 1,
        "batch_size": 1024,
        "datasets": ["qm9"],
        "start_model": "~/sciml_bench/outputs/hydronet/IPU-minima-best_model.pt",
        "load_model": False,
        "load_state": False,
        "num_features": 100,
        "num_interactions": 4,
        "num_gaussians": 25,
        "cutoff": 6.0,
        "clip_value": 200.0,
        "start_lr": 0.01,
        "loss_fn": "mse"
    }

    if 'LOCAL_RANK' in os.environ:
        # Running with torchrun
        dist.init_process_group(backend="nccl", init_method='env://')
    else:
        # Running without torchrun - force to use a single process
        dist.init_process_group(backend="nccl", rank=0, world_size=1, store=dist.HashStore())
        os.environ['LOCAL_RANK'] = '0'

    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    params_out.activate(rank=rank, local_rank=local_rank)

    console = params_out.log.console
    log = params_out.log

    console.begin('Running benchmark hydronet in training mode')
    console.message(
        f"[{os.getpid()}]: world_size = {dist.get_world_size()}, rank = {dist.get_rank()}, backend={dist.get_backend()}"
    )

    with log.subproc('Parsing input arguments'):
        args = params_in.bench_args.try_get_dict(default_args=default_args)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_gpus = dist.get_world_size()
    local_batch_size = args['batch_size']
    global_batch_size = n_gpus * local_batch_size
    # Linear LR scaling
    learning_rate = args['start_lr'] * n_gpus

    net = models_ddp.load_model_ddp(args, local_rank, device)

    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.8, min_lr=0.000001)
    
    train_loader, val_loader, train_sampler = data_ddp.init_dataloader(args, log, local_batch_size)
   
    start_time = time.time()
    for epoch in range(args['n_epochs']):
        log.begin(f'Epoch {epoch}')
        net.train()
        # let all processes sync up before starting with a new epoch of training
        dist.barrier()
       
        # DistributedSampler deterministically shuffle data
        # by seting random seed be current number epoch
        # so if do not call set_epoch when start of one epoch
        # the order of shuffled data will be always same
        train_sampler.set_epoch(epoch)
        
        with log.subproc('Training'):
            train_ddp.train_energy_only_ddp(args, log, local_rank, net, train_loader, optimizer, device)

        dist.barrier()
        
        with log.subproc('Validation'):
            val_loss = train_ddp.get_pred_eloss_ddp(args, local_rank, net, val_loader, optimizer, device)
        
        scheduler.step(val_loss) 
        dist.barrier()
        log.ended(f'Epoch {epoch}')

    end_time = time.time()
    elapsed_time = end_time - start_time

    if rank == 0:
        # Save model
        model_file = params_in.output_dir / 'hydronet_model.pt'
        log.message(f'Saving training model to {model_file}')
        torch.save(net.module.state_dict(), model_file)

        # Save metrics
        metrics = {}
        metrics['loss'] = val_loss
        metrics['time'] = elapsed_time
        metrics = {key: float(value) for key, value in metrics.items()}
        metrics_file = params_in.output_dir / 'metrics.yml'

        log.message('Saving training metrics to a file')
        with open(metrics_file, 'w') as handle:
            yaml.dump(metrics, handle)  

    dist.destroy_process_group()
    torch.cuda.synchronize()

    console.ended('Running benchmark hydronet in training mode')


def sciml_bench_inference(params_in: RuntimeIn, params_out: RuntimeOut):
    """
    Entry point for the inference routine to be called by SciML-Bench
    """

    default_args = {
        "parallel": True,
        "create_splits": False,
        "n_train": 107108,
        "n_val": 13388,
        "splitdir": "~/sciml_bench/datasets/hydronet_ds1/QM9/",
        "datadir": "sciml_bench/datasets/hydronet_ds1/",
        "inputFile" : "qm9.hdf5",
        "savedir": "trainingResults",
        "train_forces": False,
        "energy_coeff": 1,
        "n_epochs": 1,
        "batch_size": 1024,
        "datasets": ["qm9"],
        "start_model": "~/sciml_bench/outputs/hydronet/IPU-minima-best_model.pt",
        "load_model": False,
        "load_state": False,
        "num_features": 100,
        "num_interactions": 4,
        "num_gaussians": 25,
        "cutoff": 6.0,
        "clip_value": 200.0,
        "start_lr": 0.01,
        "loss_fn": "mse"
    }

    params_out.activate(rank=0, local_rank=0)

    log = params_out.log
    log.begin(f'Running benchmark hydronet on inference mode')

    with log.subproc('Parsing input arguments'):
        args = params_in.bench_args.try_get_dict(default_args=default_args)

    trainResultsDir = params_in.output_dir

    args['savedir'] = trainResultsDir

    # check for GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load trained model
    net = models_ddp.load_model(args, mode='eval', device=device, frozen=True)
    net.load_state_dict(torch.load(params_in.model))

    # load test data
    data = data_ddp.init_dataloader_from_file(args, log, 'test')
    batch_size = args['batch_size'] if len(data) > args['batch_size'] else len(data)
    loader = DataLoader(data, batch_size=batch_size, shuffle=False)

    # get predictions on test set for each dataset
    df = pd.DataFrame()
    for dataset in args['datasets']: 
        with log.subproc('Running inference'):
            start_inference_time = time.time() 
            tmp = infer(loader, net, forces=args['train_forces'], device=device)
            total_inference_time = time.time() - start_inference_time

        tmp['dataset']=dataset
        df = pd.concat([df, tmp], ignore_index=True, sort=False)
    
    # inference stats
    number_of_rows = len(df)
    throughput = total_inference_time/number_of_rows

    # save inference results
    df.to_csv(op.join(args['savedir'], 'test_set_inference.csv'), index=False)

    # Save metrics
    metrics = dict(mae=df['error'].mean(), time=total_inference_time, throughput=throughput)
    metrics['loss'] = ((df['e_actual'] - df['e_pred'])**2).sum() / len(df)
    metrics = {key: float(value) for key, value in metrics.items()}
    metrics_file = params_in.output_dir / 'metrics.yml'
    with log.subproc('Saving inference metrics to a file'):
        with open(metrics_file, 'w') as handle:
            yaml.dump(metrics, handle)  
    
    # End top level
    log.ended('Running benchmark hydronet on inference mode')

def infer(loader, net, forces=True, energies=True, force_type='max', device='cpu'):
    f_actual = []
    e_actual = []
    e_pred = []
    f_pred = []
    size = []

    for data in loader:
        data = data.to(device)
        #size += data.size.tolist()
        size += data['size'].tolist()
        if energies:
            e_actual += data.y.tolist()
            e = net(data)
            e_pred += e.tolist()

        # get predicted values
        if forces:
            data.pos.requires_grad = True
            f = torch.autograd.grad(e, data.pos, grad_outputs=torch.ones_like(e), retain_graph=False)[0]
        
            if force_type == 'max':
                f_actual += get_max_forces(data, data.f)
                f_pred += get_max_forces(data, f)
            elif force_type == 'all':
                f_actual += data.f.squeeze().tolist()
                f_pred += f.squeeze().tolist()
            elif force_type == 'error':
                f_actual += force_magnitude_error(data.f, f)
                f_pred += force_angular_error(data.f, f)
            else:
                raise ValueError('Not implemented')
                
    # return as dataframe
    if energies == True and forces == False:
 
        # Relative difference between e_actual and e_pred
        num_e_actual = np.array(e_actual)
        num_e_pred = np.array(e_pred)
        num_error = abs((num_e_actual - num_e_pred)/num_e_actual)
        error = np.ndarray.tolist(num_error)
        
        return pd.DataFrame({'size': size, 'e_actual': e_actual, 'e_pred': e_pred, 'error': error})
    
    if energies == True and forces == True:
        if force_type=='error':
            return pd.DataFrame({'size': size, 'e_actual': e_actual, 'e_pred': e_pred}), pd.DataFrame({'f_mag_error': f_actual, 'f_ang_error': f_pred})
        else:
            return pd.DataFrame({'size': size, 'e_actual': e_actual, 'e_pred': e_pred}), pd.DataFrame({'f_actual': f_actual, 'f_pred': f_pred})


# Inference methods
#
def force_magnitude_error(actual, pred):
    # ||f_hat|| - ||f||
    return torch.sub(torch.norm(pred, dim=1), torch.norm(actual, dim=1))

def force_angular_error(actual, pred):
    # cos^-1( f_hat/||f_hat|| • f/||f|| ) / pi
    # batched dot product obtained with torch.bmm(A.view(-1, 1, 3), B.view(-1, 3, 1))
    torch.pi = torch.acos(torch.zeros(1)).item() * 2 
    
    a = torch.norm(actual, dim=1)
    p = torch.norm(pred, dim=1)
    
    return torch.div(torch.acos(torch.bmm(torch.div(actual.T, a).T.view(-1, 1, 3), torch.div(pred.T, p).T.view(-1, 3,1 )).view(-1)), torch.pi)

# get_max_forces
def get_max_forces(data, forces):
    # data: data from DataLoader
    # forces: data.f for actual, f for pred
    start = 0
    f=[]
    for size in data.size.numpy()*3:
        f.append(np.abs(forces[start:start+size].numpy()).max())
        start += size
    return f