from pathlib import Path
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from PIL import Image

IMAGE_SHAPE = (200, 200, 1)

def normalize(x):
    x = (x - np.min(x))  / (np.max(x) - np.min(x))
    x = np.where(np.isnan(x), np.zeros_like(x), x)
    return x

def load_images(file_path):
    # List all TIFF files in the directory
    file_names = list(Path(file_path).glob('*.TIFF'))

    images = np.zeros((len(file_names), *IMAGE_SHAPE))
    for index, file_name in enumerate(tqdm(file_names)):
        img = Image.open(file_name)

        # A numpy array containing the tiff data
        image = np.array(img)
        image = image.astype(np.float32)
        image = normalize(image)

        # crop image around optic
        image = image[150:350, 270:470]

        image = np.expand_dims(image, axis=-1)
        images[index] = image

    return images
