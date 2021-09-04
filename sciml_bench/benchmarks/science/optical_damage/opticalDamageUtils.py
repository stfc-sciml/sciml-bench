import numpy as np
import tensorflow as tf
from pathlib import Path

import PIL
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import sys
from sys import getsizeof
import glob
from matplotlib import pyplot

import gc
import resource

import time
import yaml
import torch
import math
from pathlib import Path

def normalize(x):
    x = (x - np.min(x)) / \
        (np.max(x) - np.min(x))
    x = np.where(np.isnan(x), np.zeros_like(x), x)
    return x

def _crop_bb(self, x):
        return tf.image.crop_to_bounding_box(x, 150, 270, 200, 200)

def ssim_loss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))

# Read files from source folder, process files and store in destination folder
# The quality of the image can be "damaged" or "undamaged"

def processImages(source, destination, quality):
    # List all TIFF files in the directory
    sourceList = np.array(glob.glob(str(source) + '/*.TIFF'))
    indx = 0
    for item in sourceList:
        # Process each individual file
        img = Image.open(item)

        # A numpy array containing the tiff data
        image = np.array(img)
        image = image.astype(np.float32)
        image = normalize(image)
        image = np.expand_dims(image, axis=-1)

        #crop image
        croppedImage = tf.image.crop_to_bounding_box(image, 150, 270, 200, 200)
        
        # Cast tensorflow format to numpy.array format
        croppedImage = croppedImage.numpy()

        #save image in npy format
        filename = '/' + quality + str(indx) + '.npy'
        target = str(destination) + filename

        np.save(target, croppedImage)
        indx = indx + 1

    # number of files processed
    destinationList = np.array(glob.glob(str(destination) + '/*.npy'))
    return destinationList
