from pathlib import Path

import torchvision.transforms as T
import torch
import matplotlib.pyplot as plt


transform = T.ToPILImage()


def save_picture(*args: list[torch.Tensor], path: Path = "test.png", title: str = None):
    figure, axs = plt.subplots(len(args), len(args[0]))
    figure.set_figwidth(len(args[0]) * 4 + 2)
    figure.set_figheight(len(args) * 3 + 2)

    if title is not None:
        figure.suptitle(title)
    # figure.subplots_adjust(wspace=0, hspace=0)

    for i in range(len(args)):
        for j in range(len(args[0])):
            tensor = args[i][j]
            if tensor.dtype != torch.bool:
                image = transform(tensor)
            else:
                image = tensor.numpy().squeeze()
            axs[i][j].imshow(image, aspect="auto")
            axs[i][j].axis("off")
    figure.savefig(path)
