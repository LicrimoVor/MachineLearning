from torchvision import models
from torch import nn, load


def create_resnet18(n_features: int):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, n_features)
    return model


def load_resnet18(n_features: int, path: str):
    model = models.resnet18()
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, n_features)

    model.load_state_dict(load(path, weights_only=True))
    model.eval()
    return model


def create_resnet34(n_features: int):
    model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, n_features)
    return model


def load_resnet34(n_features: int, path: str):
    model = models.resnet34()
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, n_features)

    model.load_state_dict(load(path, weights_only=True))
    model.eval()
    return model


def create_resnet50(n_features: int):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, n_features)
    return model


def load_resnet50(n_features: int, path: str):
    model = models.resnet50()
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, n_features)

    model.load_state_dict(load(path, weights_only=True))
    model.eval()
    return model
