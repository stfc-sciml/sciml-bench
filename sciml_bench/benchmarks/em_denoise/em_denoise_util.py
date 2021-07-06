

from pathlib import Path

from mxnet import gluon
from sciml_bench.benchmarks.em_denoise.em_denoise_net import EMDenoiseNet
import yaml
import h5py
import mxnet as mx
import numpy as np
from sciml_bench.core.utils import MultiLevelLogger

# create datasets


def create_data_iter(file_noise, file_clean, batch_size):
    """ Create data iterator """
    h5_noise = h5py.File(file_noise, 'r')['images']
    h5_clean = h5py.File(file_clean, 'r')['images']
    return mx.io.NDArrayIter([h5_noise, h5_clean], batch_size=batch_size)


def create_emdenoise_datasets(dataset_dir: Path, batch_size:int):
    train_iter = create_data_iter(dataset_dir / 'train/graphene_img_noise.h5',
                                        dataset_dir / 'train/graphene_img_clean.h5',
                                        batch_size)
    test_iter = create_data_iter(dataset_dir / 'test/graphene_img_noise.h5',
                                        dataset_dir / 'test/graphene_img_clean.h5',
                                        batch_size)
    return train_iter, test_iter


def save_train_history(log: MultiLevelLogger, history: dict, output_dir: Path):
    with log.subproc('Saving training history'):
        history['train_loss'] = np.array(history['train_loss']).tolist()
        history['validate_loss'] = np.array(history['validate_loss']).tolist()
        history_file = output_dir / 'training_history.yml'
        with open(history_file, 'w') as handle:
            yaml.dump(history, handle)
        log.message(f'Training history saved to:\n{history_file}')


def parse_training_arguments(bench_args: dict):
    # default arguments
    default_args = {
        # used image size = (256/decimation) x (256/decimation)
        'decimation': 4,
        'batch_size': 128,
        'epochs': 2,
        'lr': .01,
        'use_cuda': False,
        'plot_batches': 1
    }
    # replace default_args with bench_args
    args = bench_args.try_get_dict(default_args=default_args)
    if args["decimation"] not in [1, 2, 4]:
         args["decimation"] = 4

    return args

def parse_inference_arguments(bench_args: dict):
    # default arguments
    default_args = {
        # used image size = (256/decimation) x (256/decimation)
        'decimation': 4,
        'use_cuda': False,
    }
    # replace default_args with bench_args
    args = bench_args.try_get_dict(default_args=default_args)
    if args["decimation"] not in [1, 2, 4]:
         args["decimation"] = 4

    return args


def create_em_denoise_model(args, ctx, xavier_mag):
    model = EMDenoiseNet(decimation=args["decimation"])
    model.initialize(mx.init.Xavier(magnitude=xavier_mag), ctx=ctx)
    trainer = gluon.Trainer(model.collect_params(), 'adam',
                                {'learning_rate': args["lr"]})
    return model, trainer

def organise_history(history):
    history['train_loss'] = np.array(history['train_loss']).tolist()
    history['validate_loss'] = np.array(history['validate_loss']).tolist()
    return history


def transform(img):
    return img
    
def run_batch(net, data):
    results = []
    for batch in data:
        outputs = net(batch)
        results.extend([o for o in outputs.asnumpy()])
    return np.array(results)

def setup_ctx(args: dict):
    ctx = [mx.cpu()]
    message = "Using CPU"
    if args["use_gpu"]:
        if mx.test_utils.list_gpus():
            ctx = [mx.gpu()]
            message = 'Using GPU'
        else:
            message = 'GPU is unavailable! Using CPU'

    return ctx, message 