import h5py
import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder

from sciml_bench.benchmarks.dms_structure.model import DMSNet
from sciml_bench.benchmarks.dms_structure.model import learn

# libs from sciml_bench
from sciml_bench.core.runtime import RuntimeIn, RuntimeOut

def train_model(dataset, args, params_in: RuntimeIn, params_out: RuntimeOut):
    '''
    Trains a CNN to classify the DMS data
    :param dataset: the location of the dataset
    :param args: dictionary of user/environment arguments
    :param params_in: RuntimeIn instance
    :param params_out: RuntimeOut instance
    '''

    console = params_out.log.console
    device = params_out.log.device

    learning_rate = args['learning_rate']
    epochs = args['epochs']
    batch_size = args['batch_size']
    patience = args['patience']
    model_filename = params_in.output_dir / args['model_filename']
    best_validation_model = params_in.output_dir / args['validation_model_filename']
    training_history = params_in.output_dir / args['training_history']

    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    hf = h5py.File(dataset, 'r')

    img = hf['train/images'][:]
    img = np.swapaxes(img, 1, 3)
    X_train = torch.from_numpy(np.atleast_3d(img)).to(device)
    lab = np.array(hf['train/labels']).reshape(-1, 1)
    onehot_encoder = OneHotEncoder(sparse=False)
    lab = onehot_encoder.fit_transform(lab).astype(int)
    Y_train = torch.from_numpy(lab).float().to(device)

    img = hf['test/images'][:]
    img = np.swapaxes(img, 1, 3)
    X_test = torch.from_numpy(np.atleast_3d(img)).to(device)
    lab = np.array(hf['test/labels']).reshape(-1, 1)
    lab = onehot_encoder.fit_transform(lab).astype(int)
    Y_test = torch.from_numpy(lab).float().to(device)

    model = DMSNet(device=device).to(device)

# training the network
    learn(model, X_train, Y_train, X_test,
          Y_test, epochs=epochs,
          lrate=learning_rate, batch_size=batch_size, patience=patience,
          model_filename=model_filename, best_validation_model=best_validation_model,
          training_history=training_history, device=device, smlb_out=params_out)
