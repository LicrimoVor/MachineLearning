from typing import Literal

from torch import nn

from .abstract import AbstractModule


class ConvBlock(nn.Module):
    def __init__(self, inp, out, mode: Literal["encoder", "decoder"]):
        super().__init__()
        self.act = nn.ReLU()

        mid = (inp + out) // 2
        self.conv1 = nn.Conv2d(inp, inp, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(inp)
        self.conv2 = nn.Conv2d(inp, out, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm2d(out)
        # self.conv3 = nn.Conv2d(out, out, kernel_size=3, padding=1)
        self.norm3 = nn.BatchNorm2d(out)

        match mode:
            case "encoder":
                # self.pool = nn.MaxPool2d(2)
                self.pool = nn.Conv2d(out, out, kernel_size=3, stride=2, padding=1)
            case "decoder":
                # self.pool = nn.Upsample(scale_factor=2)
                self.pool = nn.ConvTranspose2d(
                    out, out, kernel_size=3, stride=2, padding=1, output_padding=1
                )

    def forward(self, x):
        out = self.act(self.norm1(self.conv1(x)))
        out = self.act(self.norm2(self.conv2(out)))
        # out = self.act(self.norm3(self.conv3(out)))
        out = self.pool(out)
        return out


class Autoencoder(AbstractModule):
    def __init__(self, dim_code: int, mode: Literal["conv", "fully"]):
        super().__init__()
        self.name = f"AE_{mode}"
        match mode:
            case "conv":
                self.encoder = nn.Sequential(
                    ConvBlock(3, 8, "encoder"),
                    ConvBlock(8, 16, "encoder"),
                    ConvBlock(16, 32, "encoder"),
                    nn.Flatten(),
                    nn.Linear(8 * 8 * 32, dim_code),
                )

                self.decoder = nn.Sequential(
                    nn.Linear(dim_code, 8 * 8 * 32),
                    nn.Unflatten(1, (32, 8, 8)),
                    ConvBlock(32, 16, "decoder"),
                    ConvBlock(16, 8, "decoder"),
                    ConvBlock(8, 3, "decoder"),
                    nn.Conv2d(3, 3, 1),
                    nn.Sigmoid(),
                )
            case "fully":
                self.encoder = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(12288, 512),
                    nn.BatchNorm1d(512),
                    nn.LeakyReLU(3e-3),
                    nn.Linear(512, 256),
                    nn.BatchNorm1d(256),
                    nn.LeakyReLU(3e-3),
                    nn.Linear(256, dim_code),
                )

                self.decoder = nn.Sequential(
                    nn.Linear(dim_code, 256),
                    nn.BatchNorm1d(256),
                    nn.LeakyReLU(3e-3),
                    nn.Linear(256, 512),
                    nn.BatchNorm1d(512),
                    nn.LeakyReLU(3e-3),
                    nn.Linear(512, 12288),
                    nn.Unflatten(1, (3, 64, 64)),
                    nn.Sigmoid(),
                )

    def forward(self, x):
        latent_code = self.encoder(x)
        reconstruction = self.decoder(latent_code)

        return reconstruction, latent_code
