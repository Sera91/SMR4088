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
    state: distributed.DistributedState = None #TODO: setup xFFL distributed state (distributed.DistributedState)

    # WandB setup
    wandb_run: wandb.wandb_run.Run = None #TODO: setup wandb

    # Model loading from saved model
    model: nn.Module = None #TODO: load the given CNN into the given device (state.current_device)

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
        "train": None #TODO: load MNIST training set; it is available at "/leonardo/pub/userexternal/gmittone/data",
        "test": None #TODO: load MNIST test set; it is available at "/leonardo/pub/userexternal/gmittone/data",
    }

    # Dataloaders creation
    dataloaders: Dict[str, DataLoader] = {}
    for split, dataset in xffl_datasets.items():

        dataloaders[split] = None # TODO: create the dataloaders for "dataset"

        if state.rank == 0:
            logger.debug(
                f"{split} dataloader size: {len(dataloaders[split])} mini-batches"
            )

    # Optimizer and lr scheduler creation
    optimizer = None # TODO: create an optimizer

    # Main training function
    results = None #TODO: run xFFL distributed training (processing.distributed_training())

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
