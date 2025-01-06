from pathlib import Path

import torchvision.transforms as T
import torch
import matplotlib.pyplot as plt


transform = T.ToPILImage()


def save_picture(
    preds: list[list[torch.Tensor]],
    imgs: list[list[torch.Tensor]],
    path: Path = "test.png",
    title: str = None,
):
    figure, axs = plt.subplots(len(preds) * 2, len(preds[0]))
    figure.set_figheight(len(preds) * 6 + 2)
    figure.set_figwidth(len(preds[0]) * 4 + 2)

    if title is not None:
        figure.suptitle(title)
    # figure.subplots_adjust(wspace=0, hspace=0)

    for i in range(len(preds)):
        i_1 = i * 2
        i_2 = i * 2 + 1

        for j in range(len(preds[0])):

            pred = transform(preds[i][j])
            img = transform(imgs[i][j])

            axs[i_1][j].imshow(img, aspect="auto")
            axs[i_1][j].axis("off")
            axs[i_2][j].imshow(pred, aspect="auto")
            axs[i_2][j].axis("off")
    figure.savefig(path)
