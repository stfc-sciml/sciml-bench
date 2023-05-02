#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# dms_inference_dataset.py

# SciML-Bench
# Copyright © 2021 Scientific Machine Learning Research Group
# Scientific Computing Department, Rutherford Appleton Laboratory
# Science and Technology Facilities Council, UK. 
# All rights reserved.

import numpy as np
from pathlib import Path
import skimage.io
import torch
from torch.utils.data import Dataset

from sciml_bench.core.utils import list_files

class DMSInferenceDataset(Dataset):
    """ Define inference dataset for DMS """

    def __init__(self, inference_file_path: Path):
        self.inference_file_path = inference_file_path
        self.inference_file_names =  list_files(path = self.inference_file_path, recursive=True)
        self.inference_images = None
        self.inference_labels = None
        self.dataset_len = len(self.inference_file_names)
        self.initialised = False

    def __len__(self):
        return self.dataset_len
    
    def __getitem__(self, index):
        # if self.inference_images == None or self.inference_labels == None:
        if self.initialised == False: 
            self.initialised = True
            # images =  np.zeros([ len(self.inference_file_names), 1679, 1475, 3])
            images =  np.zeros([ len(self.inference_file_names), 487, 195, 3], dtype = float)
            self.inference_labels = np.zeros([len(self.inference_file_names),1], dtype = int)
            for idx, url in enumerate(self.inference_file_names):
                # images[idx, :, :, 0] =  skimage.io.imread(url)
                images[idx, :, :, :] =  skimage.io.imread(url)
                self.inference_labels[idx,0] = 0 if 'monoclinic' in str(url) else 1
            self.inference_images = torch.from_numpy( images.transpose((0, 3,1,2)) )
        
        
        return self.inference_images[index], self.inference_labels[index]

