import h5py
from pathlib import Path
import numpy as np
import tensorflow as tf

import horovod.tensorflow.keras as hvd
from .data_loader import SLSTRDataLoader
from .constants import CROP_SIZE, PATCH_SIZE, N_CHANNELS, IMAGE_H, IMAGE_W
from sciml_bench.core.runtime import RuntimeOut, RuntimeIn


def reconstruct_from_patches(patches: tf.Tensor, nx: int, ny: int, patch_size: int = PATCH_SIZE) -> tf.Tensor:
    """Reconstruct a full image from a series of patches

    :param patches: array with shape (num patches, height, width)
    :param nx: the number of patches in the x direction
    :param ny: the number of patches in the y direction
    :param patch_size: the size of th patches
    :return: the reconstructed image with shape (1, height, weight, 1)
    """
    h = ny * patch_size
    w = nx * patch_size
    reconstructed = np.zeros((1, h, w, 1))

    for i in range(ny):
        for j in range(nx):
            py = i * patch_size
            px = j * patch_size
            reconstructed[0, py:py + patch_size, px:px + patch_size] = patches[0, i, j]

    # Crop off the additional padding
    offset_y = (h - IMAGE_H) // 2
    offset_x = (w - IMAGE_W) // 2
    reconstructed = tf.image.crop_to_bounding_box(reconstructed, offset_y, offset_x, IMAGE_H, IMAGE_W)

    return reconstructed


def inference(params_in: RuntimeIn, params_out: RuntimeOut)-> None:
    """
    Main entry of `sciml_bench run` for a benchmark instance 
    in the inference mode.

    :param params_in: runtime input of `sciml_bench run`, useful components:
        * params_in.start_time: start time of running as UTC-datetime
        * params_in.dataset_dir: dataset directory for inference
        * params_in.model: model file to use 
        * params_in.output_dir: output directory
        * params_in.bench_args: benchmark-specific arguments
    :param params_out: runtime output of `sciml_bench run`, useful components:
        * params_out.log.console: multi-level logger on root (rank=0)
        * params_out.log.host: multi-level logger on host (local_rank=0)
        * params_out.log.device: multi-level logger on device (rank=any)
        * params_out.system: a set of system monitors
    """

    console = params_out.log.console
    device = params_out.log.device

    output_dir = Path(params_in.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    console.message('Loading model {}'.format(params_in.model))
    assert Path(params_in.model).exists(), "Model file does not exist!"
    model = hvd.load_model(str(params_in.model))

    console.message('Getting file paths')
    file_paths = list(Path(params_in.dataset_dir).glob('**/S3A*.hdf'))
    assert len(file_paths) > 0, "Could not find any HDF files!"

    console.message('Preparing data loader')
    # Create data loader in single image mode. This turns off shuffling and
    # only yields batches of images for a single image at a time so they can be
    # reconstructed.
    data_loader = SLSTRDataLoader(file_paths, single_image=True, crop_size=CROP_SIZE)
    dataset = data_loader.to_dataset()

    console.begin('Inference Loop')
    for patches, file_name in dataset:
        file_name = Path(file_name.numpy().decode('utf-8'))
        device.message(f"Processing file {file_name}")
        console.message(f"Processing file {file_name}")

        # convert patches to a batch of patches
        n, ny, nx, _ = patches.shape
        patches = tf.reshape(patches, (n * nx * ny, PATCH_SIZE, PATCH_SIZE, N_CHANNELS))

        # perform inference on patches
        mask_patches = model.predict_on_batch(patches)

        # crop edge artifacts
        mask_patches = tf.image.crop_to_bounding_box(mask_patches, CROP_SIZE // 2, CROP_SIZE // 2, PATCH_SIZE - CROP_SIZE, PATCH_SIZE - CROP_SIZE)

        # reconstruct patches back to full size image
        mask_patches = tf.reshape(mask_patches, (n, ny, nx, PATCH_SIZE - CROP_SIZE, PATCH_SIZE - CROP_SIZE, 1))
        mask = reconstruct_from_patches(mask_patches, nx, ny, patch_size=PATCH_SIZE - CROP_SIZE)
        mask = mask.numpy()
        mask = (mask > .5).astype(int)

        mask_name = (output_dir / file_name.name).with_suffix('.h5')

        with h5py.File(mask_name, 'w') as handle:
            handle.create_dataset('mask', data=mask)
            handle.create_dataset('mask_patches', data=mask_patches)
            handle.create_dataset('patches', data=patches)

    console.ended('Inference Loop')

