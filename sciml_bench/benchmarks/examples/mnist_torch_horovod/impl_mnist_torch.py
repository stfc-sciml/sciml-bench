#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# impl_mnist_torch.py

# SciML-Bench
# Copyright © 2021 Scientific Machine Learning Research Group
# Scientific Computing Department, Rutherford Appleton Laboratory
# Science and Technology Facilities Council, UK. 
# All rights reserved.

"""
Implementation of the MNIST_torch benchmark
"""

import h5py
import torch
import torch.utils.data as Data
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F


class MNISTDataset(Data.Dataset):
    """ mnist dataset """

    def __init__(self, file_path):
        super(MNISTDataset, self).__init__()
        # read whole
        with h5py.File(file_path, 'r') as h5:
            self.images = torch.tensor(h5['image'][:], dtype=torch.float32)
            self.labels = torch.tensor(h5['label'][:], dtype=torch.long)
        # expand channel dim and normalize
        self.images = torch.unsqueeze(self.images, 1) / 255

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.images.size(0)


def create_dataset_sampler_loader(file_path, cuda, batch_size, hvd):
    """ Create dataset, sampler and loader """
    # When supported, use 'forkserver' to spawn dataloader workers
    # instead of 'fork' to prevent issues with Infiniband implementations
    # that are not fork-safe.
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    if (kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context')
            and 'forkserver' in mp.get_all_start_methods()):
        kwargs['multiprocessing_context'] = 'forkserver'

    # create dataset
    dataset = MNISTDataset(file_path)
    # Horovod: use DistributedSampler to partition the training data
    sampler = Data.distributed.DistributedSampler(
        dataset, num_replicas=hvd.size(), rank=hvd.rank())
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, **kwargs)
    return dataset, sampler, loader


class MNISTNet(nn.Module):
    """ Define a CNN """

    def __init__(self, n_filters, n_units_hidden):
        """ Initialize network with some hyperparameters """
        super(MNISTNet, self).__init__()
        self.conv = nn.Conv2d(1, n_filters, kernel_size=3)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(n_filters * 13 * 13, n_units_hidden)
        self.fc2 = nn.Linear(n_units_hidden, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv(x)))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        x = F.softmax(self.fc2(x), dim=1)
        return x


def compute_acc(pred, truth):
    """ Compute accuracy """
    return pred.max(dim=1)[1].eq(truth).sum() / len(pred)


def metric_average(val, name, hvd):
    """ Average metric cross ranks """
    tensor = val.clone().detach()
    avg_tensor = hvd.allreduce(tensor, name=name)
    return avg_tensor.item()


def train(model, train_sampler, train_loader, test_sampler, test_loader,
          use_cuda, epochs, loss_func, optimizer_name, lr,
          batch_log_interval, hvd, smlb_out):
    """ Train model with data and hyperparameters """
    console = smlb_out.log.console
    device = smlb_out.log.device

    # training history (validation only)
    with console.subproc('Creating training history'):
        loss_val_hist = torch.zeros(epochs, dtype=torch.float32)
        acc_val_hist = torch.zeros(epochs, dtype=torch.float32)

    # send to device
    if use_cuda:
        with console.subproc('Sending model and history to device'):
            model.cuda()
            loss_val_hist = loss_val_hist.cuda()
            acc_val_hist = acc_val_hist.cuda()

    # loss
    with console.subproc('Creating loss function'):
        loss_func = eval(f'nn.{loss_func}()')
        console.message(f'Loss function: {loss_func}')

    # optimizer
    with console.subproc('Creating optimizer'):
        console.message(f'Learning rate specified: {lr}')
        console.message(f'Reduction operation: {"hvd.Average"}')
        console.message(f'Learning rate will be scaled by a factor of '
                        f'{hvd.size()} (hvd.size())')
        optimizer = eval(f'torch.optim.{optimizer_name}(model.parameters(), '
                         f'lr={lr * hvd.size()})')
        console.message(f'Optimizer: {optimizer}')
        # Horovod: wrap optimizer with DistributedOptimizer
        optimizer = hvd.DistributedOptimizer(optimizer, op=hvd.Average)

    # Horovod: broadcast model and optimizer
    with console.subproc('Broadcasting model and optimizer'):
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    # ----------------------
    # Epoch loop starts here
    # ----------------------
    console.begin('**** EPOCH LOOP ****')
    device.begin('**** EPOCH LOOP ****')
    for epoch in range(epochs):
        # only log on device within epoch loop
        device.begin(f'Epoch: {epoch}')

        # -------------------
        # Training batch loop
        # -------------------
        device.begin('Training batch loop')
        # stamp train epoch in system monitor
        smlb_out.system.stamp_event(f'epoch {epoch}: train')
        # enter train mode
        model.train()
        # Horovod: set epoch to sampler for shuffling
        train_sampler.set_epoch(epoch)
        # batch loop
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            if use_cuda:
                batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
            # forward, loss, acc
            pred_y = model(batch_x)
            loss = loss_func(pred_y, batch_y)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_idx % batch_log_interval == 0:
                # accuracy on batch
                with torch.no_grad():
                    acc = compute_acc(pred_y, batch_y)
                # Horovod: use train_sampler to determine the number of
                #          samples in this worker's partition
                device.message(
                    '[{:5d}/{:5d} ({:3.0f}%)] loss={:f}, acc={:f}, '
                    'elapsed={:f} sec'.format(
                        batch_idx * len(batch_x), len(train_sampler),
                        100 * batch_idx / len(train_loader), loss, acc,
                        device.elapsed_shallowest))
        device.ended('Training batch loop')

        # ----------------------
        # Validation on test set
        # ----------------------
        device.begin('Validating on test set')
        # stamp validate epoch in system monitor
        smlb_out.system.stamp_event(f'epoch {epoch}: validate')
        # enter eval mode
        model.eval()
        # accumulate loss and acc
        loss_val = torch.zeros((1,), dtype=torch.float32)
        acc_val = torch.zeros((1,), dtype=torch.float32)
        if use_cuda:
            loss_val, acc_val = loss_val.cuda(), acc_val.cuda()
        for batch_x, batch_y in test_loader:
            if use_cuda:
                batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
            # forward, loss, acc
            with torch.no_grad():
                pred_y = model(batch_x)
                loss_val += loss_func(pred_y, batch_y)
                acc_val += compute_acc(pred_y, batch_y) * len(pred_y)
        if use_cuda:
            loss_val, acc_val = loss_val.cpu(), acc_val.cpu()
        loss_val /= len(test_sampler)
        acc_val /= len(test_sampler)
        # average metrics across ranks and save to history
        with device.subproc('Averaging metrics across ranks (allreduce)'):
            loss_val_hist[epoch] = metric_average(loss_val, 'avg_loss', hvd)
            acc_val_hist[epoch] = metric_average(acc_val, 'avg_accuracy', hvd)
        # log device-wise and average metrics
        device.message('Metrics on rank: loss_val={:f}, acc_val={:f}'
                       .format(loss_val.item(), acc_val.item()))
        device.message('Average metrics: loss_val={:f}, acc_val={:f}'
                       .format(loss_val_hist[epoch], acc_val_hist[epoch]))
        device.ended('Validating on test set')

        # only show average on console
        console.message(f'Epoch {epoch:2d}: '
                        f'loss_val={loss_val_hist[epoch]:f}, '
                        f'acc_val={acc_val_hist[epoch]:f}, '
                        f'elapsed={device.elapsed_shallowest:f} sec')
        device.ended(f'Epoch: {epoch}')
    device.ended('**** EPOCH LOOP ****')
    console.ended('**** EPOCH LOOP ****')

    # send model and data back to CPU
    if use_cuda:
        with console.subproc('Sending model and history back to cpu'):
            model.cpu()
            loss_val_hist = loss_val_hist.cpu()
            acc_val_hist = acc_val_hist.cpu()

    # return history
    return {'loss_val': loss_val_hist.numpy().tolist(),
            'acc_val_hist': acc_val_hist.numpy().tolist()}


def predict(model, images, use_cuda, to_classes=False):
    """ Predict """
    # eval mode
    model.eval()
    # cpu => device
    if use_cuda:
        model.cuda()
        images = images.cuda()
    # compute on device
    with torch.no_grad():
        labels = model(images)
    # device => cpu
    if use_cuda:
        model.cpu()
        images = images.cpu()
        labels = labels.cpu()
    # probabilities to classes
    if to_classes:
        labels = labels.max(dim=1)[1]
    return labels
