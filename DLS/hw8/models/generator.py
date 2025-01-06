# from typing import Literal
from torch import nn
import torch

from lib.weights_init import weights_init

from .abstract import AbstractModule


class ConvBlock(nn.Module):
    def __init__(self, inp: int, out: int, kernel_size: int, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(inp, out, kernel_size, stride, padding, bias=False)
        self.norm = nn.BatchNorm2d(out)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.act(self.norm(self.conv(x)))
        return out


class ConvResizeX2(ConvBlock):
    def __init__(self, inp: int, out: int, kernel_size: int, stride=1, padding=1):
        super().__init__(inp, out, kernel_size, stride, padding)
        self.conv = nn.ConvTranspose2d(inp, out, kernel_size, stride, padding, bias=False)


class Generator(AbstractModule):
    LATENT_SIZE = 128
    IMG_SIZE = 128
    name = "generator"

    def __init__(self):
        super().__init__()

        self.main = nn.Sequential(
            ConvResizeX2(self.LATENT_SIZE, 1024, kernel_size=4, stride=1, padding=0),
            ConvBlock(1024, 1024, kernel_size=3, padding=1),
            ConvResizeX2(1024, 512, kernel_size=4, stride=2, padding=1),
            ConvBlock(512, 512, kernel_size=3, padding=1),
            ConvResizeX2(512, 256, kernel_size=4, stride=2, padding=1),
            ConvBlock(256, 256, kernel_size=3, padding=1),
            ConvResizeX2(256, 256, kernel_size=4, stride=2, padding=1),
            ConvBlock(256, 256, kernel_size=3, padding=1),
            ConvResizeX2(256, 128, kernel_size=4, stride=2, padding=1),
            ConvBlock(128, 128, kernel_size=3, padding=1),
            ConvResizeX2(128, 64, kernel_size=4, stride=2, padding=1),
            ConvBlock(64, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, 3, kernel_size=3, padding=1, bias=False),
            nn.Tanh(),
        )

        self.apply(weights_init)

    def forward(self, x):
        return self.main(x)

    def gen_latent(self, batch: int):
        return torch.randn((batch, self.LATENT_SIZE, 1, 1))
