import os
import os.path as op
from torch_geometric.data import DataListLoader, DataLoader
#from torch_geometric.loader import DataListLoader, DataLoader
from torch.utils.data import ConcatDataset
from utils.datasets import PrepackedDataset
import sys


def init_dataloader(args, 
                    split = '00', 
                    shuffle=True):
    """
    Returns train, val, and list of examine loaders
    """
    pin_memory = False if args.train_forces else True

    if not isinstance(args.datasets, list):
        args.datasets = [args.datasets]
        
    train_data = []
    val_data = []
    examine_data = []
    for ds in args.datasets:
        dataset = PrepackedDataset(None, 
                                   op.join(args.savedir,f'split_{split}_{ds}.npz'), 
                                   ds, 
                                   directory=args.datadir)
        
        train_data.append(dataset.load_data('train'))
        val_data.append(dataset.load_data('val'))

    if args.parallel:    
        train_loader = DataListLoader(ConcatDataset(train_data), batch_size=args.batch_size, shuffle=shuffle, pin_memory=pin_memory)
        val_loader = DataListLoader(ConcatDataset(val_data), batch_size=args.batch_size, shuffle=shuffle, pin_memory=pin_memory)
    else:
        train_loader = DataLoader(ConcatDataset(train_data), batch_size=args.batch_size, shuffle=shuffle, pin_memory=pin_memory)
        val_loader = DataLoader(ConcatDataset(val_data), batch_size=args.batch_size, shuffle=shuffle, pin_memory=pin_memory)
    
    return train_loader, val_loader


def test_dataloader(args, 
                    dataset,
                    split = '00'
                    ):

    dataset = PrepackedDataset(None,
                               op.join(args.savedir,f'split_{split}_{dataset}.npz'),
                               dataset,
                               directory=args.datadir)
    data = dataset.load_data('test')
    print(f"{len(data)} samples in test set")
    
    batch_size = args.batch_size if len(data) > args.batch_size else len(data)
    loader = DataLoader(data, batch_size=batch_size, shuffle=False)

    return loader


