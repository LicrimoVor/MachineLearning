import torch.nn as nn
import torch

from .abstract import AbstractModule


class EncoderBlock2(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, count_conv: int):
        super().__init__()

        self.__count_conv = count_conv
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        for i in range(count_conv - 1):
            conv = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                # nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            setattr(self, f"conv_{i+2}", conv)

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = x
        for i in range(self.__count_conv):
            out = getattr(self, f"conv_{i+1}")(out)

        self.pre_pool_out = out
        out = self.max_pool(out)
        return out


class DecoderBlock2(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, count_conv: int):
        super().__init__()

        self.__count_conv = count_conv
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        i = 0
        for _ in range(count_conv - 1):
            i += 1
            if i == 1:
                conv = nn.Sequential(
                    nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                )
            else:
                conv = nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                )
            setattr(self, f"conv_{i}", conv)

        conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        setattr(self, f"conv_{i+1}", conv)

    def forward(self, x, old_x):
        out = self.upsample(x)
        out = torch.cat((out, old_x), dim=1)
        for i in range(self.__count_conv):
            out = getattr(self, f"conv_{i+1}")(out)

        return out


class UNet(AbstractModule):

    name = "UNet"

    def __init__(self, n_class=1):
        super().__init__()
        self.encoder = nn.Sequential(
            EncoderBlock2(3, 64, 2),
            EncoderBlock2(64, 128, 2),
            EncoderBlock2(128, 256, 2),
            EncoderBlock2(256, 512, 2),
            EncoderBlock2(512, 1024, 2),
        )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=1),
            nn.Conv2d(1024, 1024, kernel_size=1),
        )

        self.decoder = nn.Sequential(
            DecoderBlock2(1024, 512, 2),
            DecoderBlock2(512, 256, 2),
            DecoderBlock2(256, 128, 2),
            DecoderBlock2(128, 64, 2),
            DecoderBlock2(64, 32, 2),
        )
        self.last_conv = nn.Conv2d(32, n_class, 1)

    def forward(self, x):
        out = x
        for block in self.encoder:
            out = block(out)

        out = self.bottleneck(out)
        for i, block in enumerate(self.decoder):
            enc_block = self.encoder[-(i + 1)]
            out = block(out, enc_block.pre_pool_out)

        out = self.last_conv(out)
        return out
