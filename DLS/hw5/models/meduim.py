from typing import Literal

from torch import nn

from .abstract import AbstractModule


class ConvLayer(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        pool: Literal["max", "adaprive_avg"] = "max",
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(in_channels)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=1)
        self.batchnorm3 = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.2)
        if pool == "max":
            self.pool = nn.MaxPool2d(kernel_size=2)
        elif pool == "adaprive_avg":
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            raise Exception("Pool type is undefined")

    def forward(self, x):
        out = self.conv1(x)
        out = self.batchnorm1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.batchnorm2(out)
        out = self.act(out)

        out = self.conv3(out + x)
        out = self.batchnorm3(out)
        out = self.act(out)
        out = self.pool(out)
        return out


class MediumCnn(AbstractModule):
    name = "Medium"

    def __init__(self, n_classes: int):
        super().__init__()
        self.conv1 = ConvLayer(3, 16)
        self.conv2 = ConvLayer(16, 32)
        self.conv3 = ConvLayer(32, 64)
        self.conv4 = ConvLayer(64, 128)
        self.conv5 = ConvLayer(128, 256)
        self.conv6 = ConvLayer(256, 512)
        self.conv7 = ConvLayer(512, 512, pool="adaprive_avg")

        self.flatten = nn.Flatten()
        # self.drop = nn.Dropout1d(0.25)
        self.act = nn.Tanh()

        self.fc1 = nn.Sequential(
            nn.Linear(512 * 1 * 1, 256),
            nn.BatchNorm1d(256),
            self.act,
        )
        self.fc2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            self.act,
        )
        self.out = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        logits = self.out(x)
        return logits
