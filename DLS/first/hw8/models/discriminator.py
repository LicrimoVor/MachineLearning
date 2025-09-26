# from typing import Literal
from torch import nn

from lib.weights_init import weights_init
from .abstract import AbstractModule


class ConvBlock(nn.Module):
    def __init__(self, inp: int, out: int, kernel_size: int, stride=1, padding=1):
        super().__init__()
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.conv = nn.Conv2d(inp, out, kernel_size, stride, padding, bias=False)
        self.norm = nn.BatchNorm2d(out)

    def forward(self, x):
        out = self.act(self.norm(self.conv(x)))
        return out


class Discriminator(AbstractModule):
    name = "discriminator"

    def __init__(self):
        super().__init__()

        self.main = nn.Sequential(
            ConvBlock(3, 64, kernel_size=3, stride=2, padding=1),  # 128 -> 64
            ConvBlock(64, 64, kernel_size=3, stride=1, padding=1),
            ConvBlock(64, 128, kernel_size=3, stride=2, padding=1),  # 64 -> 32
            ConvBlock(128, 128, kernel_size=3, stride=1, padding=1),
            ConvBlock(128, 256, kernel_size=3, stride=2, padding=1),  # 32 -> 16
            ConvBlock(256, 256, kernel_size=3, stride=1, padding=1),
            ConvBlock(256, 512, kernel_size=3, stride=2, padding=1),  # 16 -> 8
            ConvBlock(512, 1024, kernel_size=3, stride=1, padding=1),
            ConvBlock(1024, 1024, kernel_size=3, stride=2, padding=1),  # 8 -> 4
            ConvBlock(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(1024, 1, kernel_size=4, stride=2, padding=0, bias=False),  # 4 -> 1
            nn.Flatten(),
            nn.Sigmoid(),
        )

        self.apply(weights_init)

    def forward(self, x):
        return self.main(x)
