#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# dms_train.py

# SciML-Bench
# Copyright © 2021 Scientific Machine Learning Research Group
# Scientific Computing Department, Rutherford Appleton Laboratory
# Science and Technology Facilities Council, UK.
# All rights reserved.


import numpy as np
import torch
import torch
import numpy as np
from sklearn.metrics import accuracy_score

from sciml_bench.core.utils import MultiLevelLogger
from sciml_bench.core.runtime import RuntimeIn, RuntimeOut

from sciml_bench.benchmarks.dms_structure.dms_model import DMSNet


# training the neural network
def train(X_train, Y_train, batch_size, model, criterion, optimizer):
    model.train()
    train_err = 0
    train_acc = []
    for k in range(batch_size, X_train.shape[0], batch_size):
        preds = model(X_train[k-batch_size:k])
        loss = criterion(preds, Y_train[k-batch_size:k])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_err += loss.item()
        thresholded_results = np.where(
            preds.detach().cpu().numpy() > 0.5, 1, 0)
        train_acc.append(accuracy_score(thresholded_results,
                                        Y_train[k-batch_size:k].detach().cpu().numpy()))
    train_err /= (X_train.shape[0]/(batch_size))
    return train_err, np.mean(np.array(train_acc))


# Function for validation and test
def validate(X_valid, Y_valid, batch_size, model, criterion):
    valid_acc = []
    model.eval()
    with torch.no_grad():
        valid_err = 0
        for k in range(batch_size, X_valid.shape[0], batch_size):
            preds = model(X_valid[k-batch_size:k])
            valid_err += criterion(preds, Y_valid[k-batch_size:k]).item()
            thresholded_results = np.where(preds.detach().cpu().numpy()
                                           > 0.5, 1, 0)
            valid_acc.append(accuracy_score(thresholded_results,
                             Y_valid[k-batch_size:k].detach().cpu().numpy()))
    valid_err /= (X_valid.shape[0]/(batch_size))
    return valid_err, np.mean(np.array(valid_acc))


def train_model(log: MultiLevelLogger, model, datasets, args, params_in: RuntimeIn):
    """
    Trains the DMS_Structure Model
    """

    # Unpack the data first
    X_train, Y_train, X_valid, Y_valid = datasets

    # and parameters 
    learning_rate = args['learning_rate']
    epochs = args['epochs']
    batch_size = args['batch_size']
    patience = args['patience']
    #model_filename = params_in.output_dir / args['model_filename']
    #best_validation_model = params_in.output_dir / args['validation_model_filename']

    best_valid_err = np.inf
    best_train_err = np.inf
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    errors = []

    for i in range(epochs):
        train_err, train_acc = train(X_train, Y_train, batch_size, model,
                                     criterion, optimizer)
        valid_err, valid_acc = validate(X_valid, Y_valid, batch_size,
                                        model, criterion)
        if i == 0:
            patience_counter = 0
        # early stopping
        if valid_err < best_valid_err:
            #log.message(f'Saving valid current {valid_err:5.3f} best {best_valid_err:5.3f}')
            #torch.save(model.state_dict(), best_validation_model)
            best_valid_err = valid_err
            patience_counter = 0
        else:
            patience_counter += 1
        if train_err < best_train_err:
            #torch.save(model, model_filename)
            best_train_err = train_err
            #log.message('Saving model')

        if patience_counter > patience:
            log.message(
                f"Validation error did not improve in the last {patience} epochs! Stopping early.")
            break
        log.message(f"Epoch: {i}, train_err: {train_err:.4f}, train_acc: {train_acc:.4f},"
                    f"valid_err: {valid_err:.4f}, valid_acc: {valid_acc:.4f}")
        errors.append([train_err, valid_err])
    log.message(
        f"Best Results:  train_err: {best_train_err:.4f}, valid_err: {best_valid_err:.4f}")
    return np.asarray(errors)
