import torch.nn as nn

from .abstract import AbstractModule


class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, count_conv: int):
        super().__init__()

        self.__count_conv = count_conv
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        for i in range(count_conv - 1):
            conv = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            setattr(self, f"conv_{i+2}", conv)

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

    def forward(self, x):
        out = x
        for i in range(self.__count_conv):
            out = getattr(self, f"conv_{i+1}")(out)
        out, self.pool_indice = self.max_pool(out)

        return out


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, count_conv: int):
        super().__init__()

        self.__count_conv = count_conv
        self.max_unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)

        i = 0
        for _ in range(count_conv - 1):
            i += 1
            conv = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
            )
            setattr(self, f"conv_{i}", conv)

        conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        setattr(self, f"conv_{i+1}", conv)

    def forward(self, x, pool_indice):
        out = self.max_unpool(x, pool_indice)
        for i in range(self.__count_conv):
            out = getattr(self, f"conv_{i+1}")(out)

        return out


class SegNet(AbstractModule):

    name = "SegNet"

    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()

        self.encoder = nn.Sequential(
            EncoderBlock(in_channels, 64, 2),
            EncoderBlock(64, 128, 2),
            EncoderBlock(128, 256, 3),
            EncoderBlock(256, 512, 3),
        )

        self.bottleneck = nn.Conv2d(512, 512, kernel_size=1)

        self.decoder = nn.Sequential(
            DecoderBlock(512, 256, 3),
            DecoderBlock(256, 128, 3),
            DecoderBlock(128, 64, 2),
            DecoderBlock(64, out_channels, 2),
        )

    def forward(self, x):
        out = self.encoder(x)
        out = self.bottleneck(out)
        for i, block in enumerate(self.decoder):
            pool_indice = self.encoder[-(i + 1)].pool_indice
            out = block(out, pool_indice)

        return out
