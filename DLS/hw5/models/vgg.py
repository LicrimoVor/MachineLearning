from torchvision import models
from torch import nn, load


def create_vgg16(n_features: int):
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    for param in model.features.parameters():
        param.requires_grad = False

    num_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(num_features, n_features)

    return model


def load_vgg16(n_features: int, path: str):
    model = models.vgg16()
    num_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(num_features, n_features)

    model.load_state_dict(load(path, weights_only=True))
    model.eval()
    return model


def create_vgg19(n_features: int):
    model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
    for param in model.features.parameters():
        param.requires_grad = False

    num_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(num_features, n_features)

    return model


def load_vgg19(n_features: int, path: str):
    model = models.vgg19()
    num_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(num_features, n_features)

    model.load_state_dict(load(path, weights_only=True))
    model.eval()
    return model
