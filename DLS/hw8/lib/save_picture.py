from pathlib import Path

import torchvision.transforms as T
import torch
import matplotlib.pyplot as plt


transform = T.ToPILImage()


def save_picture(
    imgs: list[list[torch.Tensor]],
    preds: list[list[torch.Tensor]],
    path: Path = "test.png",
    title: str = None,
):
    figure, axs = plt.subplots(len(imgs), len(imgs[0]))
    figure.set_figheight(len(imgs) * 3 + 2)
    figure.set_figwidth(len(imgs[0]) * 2 + 2)

    if title is not None:
        figure.suptitle(title)

    for i in range(len(imgs)):

        for j in range(len(imgs[0])):
            img = transform(imgs[i][j])

            axs[i][j].imshow(img, aspect="auto")
            axs[i][j].axis("off")
            axs[i][j].set_title(f"{preds[i][j].item():.5f}")
    figure.savefig(path)
