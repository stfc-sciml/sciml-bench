import numpy as np
from pathlib import Path
import torch
from torch.utils import data
import glob

class NPZDataset(data.Dataset):
    def __init__(self, npz_root):
        self.files = glob.glob(npz_root + "/*.npz")
        print("Number of files: ", len(self.files))
   
    def __getitem__(self, index):
        sample = np.load(self.files[index])
        x = torch.from_numpy(sample["data"])
        y = sample["label"][0]
        return (x, y)
    
    def __len__(self):
        return len(self.files)
