from pathlib import Path

from torch import nn
import torch


def save_model(model: nn.Module, path: Path):
    """Сохраняет модель и возвращает путь к папке с файлами."""

    name = model.__class__.__name__
    save_path = path.joinpath(name)
    Path.mkdir(save_path)

    torch.save(
        model.state_dict(),
        save_path.joinpath("model"),
    )
    with open(save_path.joinpath("struct.txt"), "+w") as file:
        file.write(str(model))

    return save_path
