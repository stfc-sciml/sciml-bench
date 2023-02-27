import os
import numpy as np
import h5py

def create_init_split(args, suffix):
    dataset = h5py.File(os.path.join(args.datadir, f"{suffix}_data.hdf5"), "r")
    keys = list(dataset.keys())
    cluster_list = np.arange(0,dataset[keys[0]].shape[0], step=1, dtype=np.int64)
    np.random.shuffle(cluster_list)
    train_idx = cluster_list[0:args.n_train]
    val_idx = cluster_list[args.n_train:(args.n_train+args.n_val)]
    test_idx = cluster_list[(args.n_train+args.n_val):]
    np.savez(os.path.join(args.savedir, f'split_00_{suffix}.npz'),
             train_idx=np.array(train_idx), 
             val_idx=np.array(val_idx),
             test_idx=np.array(test_idx))

