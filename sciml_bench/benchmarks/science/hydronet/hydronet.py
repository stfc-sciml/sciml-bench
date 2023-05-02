import os
import time
import torch
import yaml
import pandas as pd
import os.path as op
from sciml_bench.core.runtime import RuntimeIn, RuntimeOut

import torch.distributed as dist
from torch_geometric.loader import DataLoader
from sciml_bench.benchmarks.science.hydronet.utils import data_ddp, train_ddp, models_ddp, infer

def sciml_bench_training(params_in: RuntimeIn, params_out: RuntimeOut):
    """
    Main entry of `sciml_bench run` for a benchmark instance in the training mode.
    """

    default_args = {
        "train_forces": False,
        "energy_coeff": 1,
        "n_epochs": 500,
        "batch_size": 1024,
        "dataset": "qm9",
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
        dist.init_process_group(backend="nccl")
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
        args['datadir'] = params_in.dataset_dir

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_gpus = dist.get_world_size()
    local_batch_size = args['batch_size']
    global_batch_size = n_gpus * local_batch_size
    # Linear LR scaling
    learning_rate = args['start_lr']

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
        "n_train": 107108,
        "n_val": 13388,
        "train_forces": False,
        "energy_coeff": 1,
        "n_epochs": 1,
        "batch_size": 1024,
        "dataset": "qm9",
        "load_model": True,
        "load_state": True,
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

    args['savedir'] = params_in.output_dir

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
    with log.subproc('Running inference'):
        start_inference_time = time.time() 
        df = infer.infer(loader, net, forces=args['train_forces'], device=device)
        total_inference_time = time.time() - start_inference_time

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
