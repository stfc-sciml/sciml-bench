# For running the code, these packages are required:
# pip install torch
# pip install prettytable
#
# To run the code issue this command:
# python synthetic_v1.py
#
import wandb
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import lightning.pytorch as pl
import torch.distributed as dist
from torch.utils.data import DataLoader

from sciml_bench.core.runtime import RuntimeIn, RuntimeOut


class RegressionDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples, input_size):
        super().__init__()
        X = torch.randn(num_samples, input_size)
        y = X**2 + 15 * np.sin(X) ** 3
        y_t = torch.sum(y, dim=1)
        self._x = X
        self._y = y_t

    def __len__(self):
        return self._x.shape[0]

    def __getitem__(self, index):
        x = self._x[index]
        y = self._y[index]
        return x, y


class SyntheticRegression(pl.LightningModule):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.net = torch.nn.Sequential(
            nn.Linear(input_size, hidden_size),  # input -> hidden layer
            nn.Sigmoid(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, output_size),  # hidden -> output layer
            nn.Sigmoid(),
        )

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.net(x).flatten()
        loss = F.mse_loss(y_hat, y)
        wandb.log({"train_loss": loss})
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.001)
        return optimizer


def sciml_bench_training(params_in: RuntimeIn, params_out: RuntimeOut):
    default_args = {
        "input_size": 784,
        "batch_size": 128,
        "num_samples": 1024000,
        "hidden_size": 3000,
        "epochs": 1,
    }

    wandb.init(project="synthetic_regression", config=default_args)

    if "LOCAL_RANK" in os.environ:
        # Running with torchrun
        dist.init_process_group(backend="nccl", init_method="env://")
    else:
        # Running without torchrun - force to use a single process
        dist.init_process_group(
            backend="nccl", rank=0, world_size=1, store=dist.HashStore()
        )
        os.environ["LOCAL_RANK"] = "0"

    local_rank = int(os.environ["LOCAL_RANK"])
    params_out.activate(rank=dist.get_rank(), local_rank=local_rank)

    # Log top level process
    log = params_out.log.console
    log.begin(f"Running benchmark synthetic_regression in training mode")

    # Parse input arguments against default ones
    with log.subproc("Parsing input arguments"):
        args = params_in.bench_args.try_get_dict(default_args=default_args)

    num_samples = args["num_samples"]
    batch_size = args["batch_size"]

    with log.subproc("Creating dataset"):
        dataset = RegressionDataset(num_samples, args["input_size"])
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    log.message(f"Number of samples: {num_samples}")
    log.message(f"Total number of batches: {len(dataloader)}, {num_samples/batch_size}")

    model = SyntheticRegression(args["input_size"], args["hidden_size"], output_size=1)

    with log.subproc("Training model"):
        trainer = pl.Trainer(
            accelerator="gpu",
            devices=dist.get_world_size(),
            strategy="ddp",
            max_epochs=args["epochs"],
            default_root_dir=params_in.output_dir,
        )
        start_time = time.time()
        trainer.fit(model, dataloader)
        end_time = time.time()

    time_taken = end_time - start_time

    metrics = {}
    metrics["time"] = time_taken

    wandb.log(metrics)
