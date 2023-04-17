import math
import os
import time
from typing import List
import torch
import os.path as op
from torch_geometric.data import DataLoader, Dataset, Data
from torch.utils.data import Subset
import torch.distributed as dist
import h5py
from pathlib import Path

from torch.utils.data.distributed import DistributedSampler

class HydronetDataset(Dataset):
    def __init__(self, path):
        with h5py.File(path, 'r') as handle:
            self.size = torch.from_numpy(handle['size'][:])
            self.x = torch.from_numpy(handle['x'][:])
            self.z = torch.from_numpy(handle['z'][:])
            self.pos = torch.from_numpy(handle['pos'][:])
            self.y = torch.from_numpy(handle['y'][:])

        self.n_samples = self.size.shape[0]

        self.data_list = []
        for index in range(self.n_samples):
            size = self.size[index]
            data = Data(x=self.x[index][:size], y=self.y[index], z=self.z[index][:size], pos=self.pos[index][:size], size=size)
            self.data_list.append(data)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return self.data_list[index]


# init_dataloader_from_file
def init_dataloader_from_file(args, log, actionStr, split = '00', shuffle=True):
    """
    Returns train, val, and list of examine loaders
    """
    path = os.path.join(os.path.expanduser("~"), args["datadir"])
    file_name = Path(path) / f'{args["datasets"][0]}.hdf5'
    start_time = time.time()
    dataset = HydronetDataset(file_name)
    end_time = time.time()

    log.message(f"Loaded data in {end_time - start_time:.2f}s on rank {dist.get_rank()}")
    
    # Split data 80:10:10
    n = len(dataset)
    indices = torch.arange(n)
    train_size = math.floor(n*0.8)
    val_size = math.ceil(n*0.1)

    train_dataset = Subset(dataset, indices[:train_size])
    val_dataset = Subset(dataset, indices[train_size:train_size+val_size])
    test_dataset = Subset(dataset, indices[train_size+val_size:])

    if(actionStr =='train'):
        return train_dataset, val_dataset

    if(actionStr =='test'):
        return test_dataset

#
def init_dataloader(args, log, local_batch_size):
    """
    Returns train, val, and list of examine loaders
    """
    # pin_memory = False if args.train_forces else True
    pin_memory = True
    num_workers = 0

    if not isinstance(args["datasets"], list):
        args["datasets"] = [args["datasets"]]

    trainData, valData = init_dataloader_from_file(args, log, "train")

    train_sampler = DistributedSampler(trainData, shuffle=True, drop_last=True)
    train_loader = DataLoader(trainData, sampler=train_sampler, pin_memory=pin_memory, num_workers=num_workers, batch_size=local_batch_size)
    
    val_sampler = DistributedSampler(valData, shuffle=True, drop_last=True)
    val_loader = DataLoader(valData, sampler=val_sampler, pin_memory=pin_memory, num_workers=num_workers, batch_size=local_batch_size)

    log.message("Train data size {:5d}".format(len(trainData)))
    log.message("train_loader size {:5d}".format(len(train_loader.dataset)))
    log.message("val_loader size {:5d}".format(len(val_loader.dataset)))

    return train_loader, val_loader, train_sampler


def test_dataloader(args, 
                    dataset,
                    split = '00'
                    ):

    dataset = PrepackedDataset(None, 
                               op.join(args.savedir,f'split_{split}_{dataset}.npz'), 
                               dataset, 
                               directory=args.datadir)
    data = dataset.load_data('test')
    
    batch_size = args.batch_size if len(data) > args.batch_size else len(data)
    loader = DataLoader(data, batch_size=batch_size, shuffle=False)

    return loader


