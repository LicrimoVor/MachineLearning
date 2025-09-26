from torch import nn
from .abstract import AbstractModule


class SimpleCnn(AbstractModule):
    name = "simple"

    def __init__(self, n_classes: int):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 512, kernel_size=3),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(1024, 2048, kernel_size=3),
            nn.BatchNorm2d(2048),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(2048, 4096, kernel_size=3),
            nn.BatchNorm2d(4096),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d((1, 1)),
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(4096, 4096, kernel_size=3, padding=1),
            nn.BatchNorm2d(4096),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.flatten = nn.Flatten()
        # self.drop = nn.Dropout1d(0.25)
        self.act = nn.Tanh()

        self.fc1 = nn.Sequential(
            nn.Linear(4096 * 1 * 1, 2048),
            nn.BatchNorm1d(2048),
            self.act,
        )
        self.fc2 = nn.Sequential(
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            self.act,
        )
        self.fc3 = nn.Sequential(
            nn.Linear(512, 128),
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
        x = self.fc3(x)
        logits = self.out(x)
        return logits
