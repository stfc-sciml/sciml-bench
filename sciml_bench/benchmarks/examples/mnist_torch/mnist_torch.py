#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# mnist_torch.py

# SciML-Bench
# Copyright © 2021 Scientific Machine Learning Research Group
# Scientific Computing Department, Rutherford Appleton Laboratory
# Science and Technology Facilities Council, UK.
# All rights reserved.

"""
Benchmark: mnist_torch
Classifying MNIST using a CNN implemented with PyTorch.
This is a single device training/inference example.
"""

# libs from sciml_bench
from sciml_bench.core.runtime import RuntimeIn, RuntimeOut

# libs required by implementation
import yaml
import torch
import torchmetrics as tm
import torch.nn.functional as F
import h5py
import wandb
from torch.utils.data import DataLoader

# implementation from other files
from sciml_bench.benchmarks.examples.mnist_torch.mnist_torch_misc import (
    MNISTDataset,
    MNISTNet,
)


def create_model_mnist():
    model = MNISTNet(32, 128)
    return model


def train(model, train_loader, optimizer):
    model.train()
    accuracy_metric = tm.Accuracy("multiclass", num_classes=10)
    loss_metric = tm.MeanMetric()

    for _, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)

        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        accuracy_metric(output, target)
        loss_metric(loss)

    wandb.log({"loss": loss_metric.compute(), "accuracy": accuracy_metric.compute()})


def predict(model, test_loader):
    model.eval()
    accuracy_metric = tm.Accuracy("multiclass", num_classes=10)
    loss_metric = tm.MeanMetric()

    for _, (data, target) in enumerate(test_loader):
        output = model(data)
        loss = F.cross_entropy(output, target)
        accuracy_metric(output, target)
        loss_metric(loss)

    wandb.log(
        {"val_loss": loss_metric.compute(), "val_accuracy": accuracy_metric.compute()}
    )


def sciml_bench_training(params_in: RuntimeIn, params_out: RuntimeOut):
    """
    Main entry of `sciml_bench run` for a benchmark instance
    in the training mode. Please consult the API documentation.
    """
    suggested_args = {"batch_size": 128, "epochs": 2}
    args = params_in.bench_args.try_get_dict(default_args=suggested_args)

    wandb.init(project="mnist_torch", config=args)

    params_out.activate(
        rank=0,
        local_rank=0,
        activate_log_on_host=False,
        activate_log_on_device=True,
        console_on_screen=True,
    )

    log = params_out.log
    log.begin("Running benchmark mnist_torch on training mode")

    # top-level process
    console = params_out.log.console
    console.begin("Running benchmark mnist_torch on training mode")

    # We expect two benchmark-specific arguments here:
    # batch_size and epochs. If not, we will assign
    # default values.
    with log.subproc("Parsing input arguments"):
        # hyperparameters
        suggested_args = {"batch_size": 128, "epochs": 2}
        args = params_in.bench_args.try_get_dict(default_args=suggested_args)
        batch_size = args["batch_size"]
        epochs = args["epochs"]
        log.message(f"batch_size = {batch_size}")
        log.message(f"epochs     = {epochs}")

    with log.subproc("Writing the argument file"):
        args_file = params_in.output_dir / "arguments_used.yml"
        with open(args_file, "w") as handle:
            yaml.dump(args, handle)

    # create datasets
    with log.subproc("Creating datasets"):
        dataset_dir = params_in.dataset_dir
        train_set = MNISTDataset(dataset_dir / "train.hdf5")
        test_set = MNISTDataset(dataset_dir / "test.hdf5")
        log.message(f"Dataset directory: {dataset_dir}")

    # create model
    with log.subproc("Creating CNN model"):
        model = create_model_mnist()

    # data
    console.begin("Creating data loader")
    dataset_dir = params_in.dataset_dir
    train_dataloader = DataLoader(
        train_set, batch_size, shuffle=True, num_workers=0, pin_memory=True
    )
    test_dataloader = DataLoader(
        test_set, batch_size, shuffle=False, num_workers=0, pin_memory=True
    )
    console.ended("Creating data loader")

    # Training
    log.begin("Training CNN model")
    with log.subproc("Running model.fit()"):
        params_out.system.stamp_event("start train()")
        optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
        for epoch in range(args["epochs"]):
            train(model, train_dataloader, optimizer)
            predict(model, test_dataloader)
            log.message(f"epoch: {epoch}")

    # save model
    with log.subproc("Saving the model"):
        model_file = params_in.output_dir / "mnist_torch_model.h5"
        torch.save(model.state_dict(), model_file)
        log.message(f"Saved to: {model_file}")

    log.ended("Training CNN model")

    # top-level process
    console.ended("Running benchmark mnist_torch on training mode")
