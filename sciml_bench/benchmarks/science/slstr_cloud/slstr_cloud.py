#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# slstr_cloud.py

# SciML-Bench
# Copyright © 2021 Scientific Machine Learning Research Group
# Scientific Computing Department, Rutherford Appleton Laboratory
# Science and Technology Facilities Council, UK.
# All rights reserved.

from sciml_bench.core.utils import MultiLevelLogger
import yaml
import tensorflow as tf
import horovod.tensorflow as hvd

from sciml_bench.core.runtime import RuntimeIn, RuntimeOut

from sciml_bench.benchmarks.slstr_cloud.train import train_model
from sciml_bench.benchmarks.slstr_cloud.data_loader import load_datasets
from sciml_bench.benchmarks.slstr_cloud.model import unet


# from sciml_bench.benchmarks.slstr_cloud.inference import inference


# Sets the target device
def set_target_devices(use_gpu: bool, log: MultiLevelLogger) -> bool:
    if use_gpu:
        try:
            tf.config.set_visible_devices([], 'GPU')
            gpus = tf.config.experimental.list_physical_devices('GPU')
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            if gpus:
                tf.config.experimental.set_visible_devices(
                    gpus[hvd.local_rank()], 'GPU')

            if hvd.rank() == 0:
                log.message('Using GPU(s)')
                log.message(f"Num GPUS: {len(gpus)}")
                log.message(f"Num ranks: {hvd.size()}")
            return True
        except:
            use_gpu = False


    if not use_gpu:
        try:
            tf.config.set_visible_devices([], 'CPU')
            log.message('Using CPU(s)')
            return True
        except:
            log('Failed to use CPUs! Terminating.')
            return False


#####################################################################
# Training mode                                                     #
#####################################################################


def sciml_bench_training(params_in: RuntimeIn, params_out: RuntimeOut):
    """
    Main entry for `sciml_bench run` in training mode
    """

    # initialize horovod
    hvd.init()

    # initialize monitor with hvd.rank() and hvd.local_rank()
    params_out.activate(rank=hvd.rank(), local_rank=hvd.local_rank(),
                          activate_log_on_host=False,
                          activate_log_on_device=True, console_on_screen=True)

    console = params_out.log.console

    # top-level process
    console.begin('Running benchmark slstr_cloud in training mode.')
    console.message(f'hvd.rank()={hvd.rank()}, hvd.size()={hvd.size()}')


    default_args = {
        # tensorflow env
        'seed': 1234,
        'use_gpu': True,
        # hyperparameters
        'learning_rate': 0.001,
        'epochs': 30,
        'batch_size': 32,
        'wbce': .5,
        'clip_offset': 15,
        'train_split': .8,
        'crop_size': 80,
        'no_cache': False,
    }


    with console.subproc('Parsing input arguments'):
        args = params_in.bench_args.try_get_dict(default_args)


    tf.random.set_seed(args['seed'])
    console.message(f'Random seed: {args["seed"]}')

    train_data_dir = params_in.dataset_dir / 'one-day'
    test_data_dir =  params_in.dataset_dir / 'ssts'


    if set_target_devices(args['use_gpu']) == False:
        console.ended('Running benchmark slstr_cloud in training mode.')
        return 

            
    # save arguments
    if hvd.rank() == 0:
        args_file = params_in.output_dir / 'training_arguments_used.yml'
        with console.subproc('Saving arguments to a file'):
            with open(args_file, 'w') as handle:
                yaml.dump(args, handle)

    # load the datasets
    with console.subproc("Loading datasets"):
        train_dataset, test_dataset  = load_datasets(dataset_dir=train_data_dir, args=args)
    
    # build the UNet model
    with console.subproc('Creating the model'):
        model = unet(input_shape=(256, 256, 9))
        
        
    # train model
    with console.subproc("Training the model"):
        history = train_model(train_dataset, test_dataset, model, args, params_in, params_out)


    # save history
    with console.subproc('Saving training history'):
        history_file = params_in.output_dir / 'training_history.yml'
        with history_file.open('w') as handle:
            yaml.dump(history, handle)


    console.begin('Running benchmark slstr_cloud in training mode.')






#####################################################################
# Inference mode                                                    #
#####################################################################
def sciml_bench_inference(params_in: RuntimeIn, params_out: RuntimeOut):
    """
    Main entry for `sciml_bench run` in inference mode.
    """

    # initialize horovod
    hvd.init()

    # initialize monitor with hvd.rank() and hvd.local_rank()
    params_out.activate(rank=hvd.rank(), local_rank=hvd.local_rank(),
                          activate_log_on_host=False,
                          activate_log_on_device=True, console_on_screen=True)

    console = params_out.log.console

    # top-level process
    console.begin('Running benchmark slstr_cloud in inference mode.')
    console.message(f'hvd.rank()={hvd.rank()}, hvd.size()={hvd.size()}')


    default_args = {
        'use_gpu': True,
        'crop_size': 80
    }


    with console.subproc('Parsing input arguments'):
        args = params_in.bench_args.try_get_dict(default_args)

          
    # save arguments
    if hvd.rank() == 0:
        args_file = params_in.output_dir / 'inference_arguments_used.yml'
        with console.subproc('Saving inference arguments to a file'):
            with open(args_file, 'w') as handle:
                yaml.dump(args, handle)

    # load the datasets
    with console.subproc("Loading inference datasets"):
        train_dataset, test_dataset  = load_inferece_datasets(dataset_dir=inference_data_dir, args=args)
    
    # build the UNet model
    with console.subproc('Loading the model'):
        model = hvd.load_model()
        

    with console.subproc('Preparing data loader'):
        data_loader = SLSTRDataLoader(file_paths, single_image=True, crop_size=args["crop_size"])
        dataset = data_loader.to_dataset()


    console.message('Loading model {}'.format(params_in.model_file))
    assert Path(params_in.model_file).exists(), "Model file does not exist!"
    model = hvd.load_model(str(params_in.model_file))

    console.message('Getting file paths')
    file_paths = list(Path(params_in.dataset_dir).glob('**/S3A*.hdf'))
    assert len(file_paths) > 0, "Could not find any HDF files!"

    console.message('Preparing data loader')
    # Create data loader in single image mode. This turns off shuffling and
    # only yields batches of images for a single image at a time so they can be
    # reconstructed.
    data_loader = SLSTRDataLoader(file_paths, single_image=True, crop_size=crop_size)
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
        mask_patches = tf.image.crop_to_bounding_box(mask_patches, crop_size // 2, crop_size // 2, PATCH_SIZE - crop_size, PATCH_SIZE - crop_size)

        # reconstruct patches back to full size image
        mask_patches = tf.reshape(mask_patches, (n, ny, nx, PATCH_SIZE - crop_size, PATCH_SIZE - crop_size, 1))
        mask = reconstruct_from_patches(mask_patches, nx, ny, patch_size=PATCH_SIZE - crop_size)
        mask_name = (output_dir / file_name.name).with_suffix('.h5')

        with h5py.File(mask_name, 'w') as handle:
            handle.create_dataset('mask', data=mask)

    console.ended('Inference Loop')