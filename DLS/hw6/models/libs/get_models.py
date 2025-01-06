from typing import Callable

from torch import nn

from ..simple import SimpleCnn
from ..meduim import MediumCnn
from ..resnet import create_resnet18, create_resnet34
from ..vgg import create_vgg16, create_vgg19


def get_models(n_features: int) -> list[Callable[[], nn.Module]]:

    callback_models = [
        lambda: SimpleCnn(n_features),
        lambda: MediumCnn(n_features),
        lambda: create_resnet18(n_features),
        lambda: create_resnet34(n_features),
        lambda: create_vgg16(n_features),
        lambda: create_vgg19(n_features),
    ]

    return callback_models
