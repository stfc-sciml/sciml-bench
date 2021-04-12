from sciml_bench.core.runtime import RuntimeIn, RuntimeOut
from sciml_bench.core.utils import SafeDict

import yaml
import tensorflow as tf
import horovod.tensorflow as hvd

from sciml_bench.benchmarks.slstr_cloud.train import train_model
from sciml_bench.benchmarks.slstr_cloud.inference import inference


def sciml_bench_run(smlb_in: RuntimeIn, smlb_out: RuntimeOut):
    """
    Main entry of `sciml_bench run` for a benchmark instance

    :param smlb_in: runtime input of `sciml_bench run`, useful components:
        * smlb_in.start_time: start time of running as UTC-datetime
        * smlb_in.dataset_dir: dataset directory
        * smlb_in.output_dir: output directory
        * smlb_in.bench_args: benchmark-specific arguments
    :param smlb_out: runtime output of `sciml_bench run`, useful components:
        * smlb_out.log.console: multi-level logger on root (rank=0)
        * smlb_out.log.host: multi-level logger on host (local_rank=0)
        * smlb_out.log.device: multi-level logger on device (rank=any)
        * smlb_out.system: a set of system monitors
    """


    # -----------------------------------
    # initialize horovod and smlb_monitor
    # -----------------------------------
    # initialize horovod
    hvd.init()

    # initialize smlb_monitor with hvd.rank() and hvd.local_rank()
    # In this example, we will log the whole process on console and the
    # epoch loop on device, leaving the host logger unactivated.
    smlb_out.activate(rank=hvd.rank(), local_rank=hvd.local_rank(),
                          activate_log_on_host=False,
                          activate_log_on_device=True, console_on_screen=True)

    # in this example, we will log sub-processes in console while
    # using the device logger only in epoch loop for clarity
    console = smlb_out.log.console

    # top-level process
    console.begin('Running benchmark slstr_cloud')
    console.message(f'hvd.rank()={hvd.rank()}, hvd.size()={hvd.size()}')

    # ------------------------------------------
    # input arguments and tensorflow environment
    # ------------------------------------------
    console.begin('Handling input arguments and tensorflow environment')
    # default arguments
    default_args = {
        # tensorflow env
        'seed': 1234,
        'use_cuda': True,
        # hyperparameters
        'learning_rate': 0.001,
        'epochs': 30,
        'batch_size': 32,
        'wbce': .5,
        'clip_offset': 15,
        'train_split': .8,
        'crop_size': 80,
        'no_cache': False,
        # workflow control
        'load_weights_file': '',  # do training if this is empty
    }
    args = smlb_in.bench_args.try_get_dict(default_args)

    train_data_dir = smlb_in.dataset_dir / 'one-day'
    test_data_dir =  smlb_in.dataset_dir / 'ssts'

    # tensorflow environment
    tf.random.set_seed(args['seed'])
    console.message(f'Random seed: {args["seed"]}')
    if args['use_cuda']:
        try:
            # Disable all GPUS
            tf.config.set_visible_devices([], 'GPU')
            visible_devices = tf.config.get_visible_devices()
            for device in visible_devices:
                assert device.device_type != 'GPU'
        except:
            # Invalid device or cannot modify virtual devices once initialized.
            pass

    # save actually used arguments
    if hvd.rank() == 0:
        args_file =  smlb_in.output_dir / 'arguments_used.yml'
        with open(args_file, 'w') as handle:
            yaml.dump(args, handle)
        console.message(f'Arguments used are saved to:\n{args_file}')

    console.ended('Handling input arguments and tensorflow environment')

    # Train model or load weights from file
    if args['load_weights_file'] == '':
        console.begin('Training model')
        train_model(train_data_dir, args, smlb_in, smlb_out)
        args['load_weights_file'] =  smlb_in.output_dir / 'model.h5'
        console.ended('Training model')

    # Inference
    console.begin('Inference')
    inference(args['load_weights_file'], test_data_dir, args, smlb_in, smlb_out)
    console.ended('Inference')

