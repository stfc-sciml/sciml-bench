import numpy as np
from pathlib import Path
import torch
from torch.utils import data
import glob
import argparse
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms, models
import horovod.torch as hvd
import os
import math
from npz_dataset import NPZDataset
import torch.nn as nn

class NPZDataset(data.Dataset):
    def __init__(self, npz_root):
        #self.files = glob.glob(npz_root + "/*.npz")
        files1 = glob.glob(npz_root + "/*.npz")
        self.files = files1[0:31]
        print("Number of files: ", len(self.files))
   
    def __getitem__(self, index):
        sample = np.load(self.files[index])
        x = torch.from_numpy(sample["data"])
        y = sample["label"][0]
        return (x, y)
    
    def __len__(self):
        return len(self.files)

class SGCLDataLoader:
    def __init__(self, data_dir, shuffle=True, **kwargs):
        self._data_dir = data_dir

        self._image_paths = Path(self._data_dir).glob('**/S3A*.hdf')
        self._image_paths = list(map(str, self._image_paths))

        self._shuffle = shuffle
       
        # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
        # issues with Infiniband implementations that are not fork-safe
        if (kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context') and
            mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
            kwargs['multiprocessing_context'] = 'forkserver'
        
        print("Training directory: ", data_dir)
        self.dataset = NPZDataset(data_dir)
        batch_size = kwargs["allreduce_batch_size"]
        del kwargs["allreduce_batch_size"]
        # Horovod: use DistributedSampler to partition data among workers. Manually specify
        # `num_replicas=hvd.size()` and `rank=hvd.rank()`.
        self.sampler = torch.utils.data.distributed.DistributedSampler(
            self.dataset, num_replicas=hvd.size(), rank=hvd.rank())
        self.loader = torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size,
            sampler=self.sampler, **kwargs)
   
    def get_dataset(self):
        return self.dataset
   
    def get_dataloader(self):
        return self.dataloader
