"""Simple MLP training script"""

import argparse
import sys
from logging import Logger, getLogger
from parser import parser
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.optim import Adadelta
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms

from xffl.distributed import distributed
from xffl.learning import processing, utils
from xffl.utils.logging import setup_logging

logger: Logger = getLogger(__name__)
"""Default xFFL logger"""

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        output = F.log_softmax(self.fc3(x), dim=1)
        return output


def training(args: argparse.Namespace) -> None:
    """Simple MLP training script

    :param args: Command-line arguments
    :type args: argparse.Namespace
    :param model_info: Model information class
    :type model_info: ModelInfo
    :param dataset_info: Dataset information class
    :type dataset_info: DatasetInfo
    """
    # Set the requested logging level
    setup_logging(log_level=args.loglevel)

    # Sets RNGs seeds and force PyTorch's deterministic execution
    generator: Optional[torch.Generator] = (
        utils.set_deterministic_execution(seed=args.seed) if args.seed else None
    )

    # PyTorch's distributed backend setup
    state: distributed.DistributedState = distributed.setup_distributed_process_group(
        hsdp=args.hsdp, federated=args.federated_scaling, streams=args.cuda_streams
    )

    # WandB setup
    wandb_run: wandb.wandb_run.Run = #TODO: setup wandb

    # Model loading from saved model
    model: nn.Module = #TODO: load the given CNN into the given device (state.current_device)

    # Print model's weights
    if state.rank == 0:
        logger.debug(
            f"Training a simple MLP: {(utils.get_model_size(model=model) / 1e6):.4f} million trainable parameters"
        )

    # Dataset loading
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    xffl_datasets: Dict[str, Dataset] = {
        "train": #TODO: load MNIST training set; it is available at "/leonardo/pub/userexternal/gmittone/data",
        "test": #TODO: load MNIST test set; it is available at "/leonardo/pub/userexternal/gmittone/data",
    }

    # MNIST splitting into single classes (with equal dataset size)
    if args.one_class:
        for _, dataset in xffl_datasets.items():
            dataset.data = dataset.data[dataset.targets == state.rank % 10]
            dataset.targets = dataset.targets[dataset.targets == state.rank % 10]
    xffl_datasets["train"] = Subset(xffl_datasets["train"], range(5000))
    xffl_datasets["test"] = Subset(xffl_datasets["train"], range(800))

    # Dataloaders creation
    dataloaders: Dict[str, DataLoader] = {}
    for split, dataset in xffl_datasets.items():

        dataloaders[split] = # TODO: create the dataloaders for "dataset"

        if state.rank == 0:
            logger.debug(
                f"{split} dataloader size: {len(dataloaders[split])} mini-batches"
            )

    # Optimizer and lr scheduler creation
    optimizer = # TODO: create an optimizer

    # Main training function
    results = processing.distributed_training(
        model=model,
        state=state,
        optimizer=optimizer,
        train_dataloader=dataloaders["train"],
        eval_dataloader=dataloaders["test"],
        wandb_run=wandb_run,
        epochs=args.epochs,
        federated_batches=args.federated_batches,
        criterion=nn.NLLLoss(),
    )

    if state.rank == 0:
        [logger.debug(f"Key: {k}, Value: {v}") for k, v in results.items()]
        if args.wandb:
            for k, v in results.items():
                wandb_run.summary[k] = v

    # PyTorch's distributed backend cleanup
    wandb.finish()
    distributed.cleanup_distributed_process_group(state=state)


def main():
    """Argument parsing and training launch"""

    try:
        args = parser.parse_args(sys.argv[1:])
        training(args=args)
    except KeyboardInterrupt as e:
        logger.exception(e)
    except Exception as e:
        logger.exception(e)


if __name__ == "__main__":
    main()
