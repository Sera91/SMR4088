"""List of already supported models and datasets"""

from typing import Final, Mapping

from .datasets.clean_mc4_it import CleanMc4It
from .datasets.dataset_info import DatasetInfo
from .models.llama import Llama
from .models.mixtral import Mixtral
from .models.model_info import ModelInfo

MODELS: Final[Mapping[str, ModelInfo]] = {
    "tiny-random-llama-3": Llama(
        path="/leonardo/pub/userexternal/gmittone/models/tiny-random-llama-3"
    ),
    "llama3.1-8b": Llama(path="/leonardo/pub/userexternal/gmittone/models/llama3.1-8b"),
    "llama3.1-70b": Llama(
        path="/leonardo/pub/userexternal/gmittone/models/llama3.1-70b"
    ),
    "mixtral-8x7b-v0.1": Mixtral(
        path="/leonardo/pub/userexternal/gmittone/models/mixtral-8x7b-v0.1"
    ),
}
""" Supported models dictionary """

DATASETS: Final[Mapping[str, DatasetInfo]] = {
    "clean_mc4_it": CleanMc4It(
        path="/leonardo/pub/userexternal/gmittone/datasets/clean_mc4_it"
    ),
}
""" Supported datasets dictionary """
