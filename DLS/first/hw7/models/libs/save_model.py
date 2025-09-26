from datetime import datetime as dt
from pathlib import Path

from torch import nn
import torch


BASE_DIR = Path(__file__).parent.parent


def save_model(model: nn.Module):
    """Сохраняет модель и возвращает путь к папке с файлами."""

    name = model.__class__.__name__
    if not BASE_DIR.joinpath(f"{name}").exists():
        BASE_DIR.joinpath(f"{name}").mkdir()

    save_path = BASE_DIR.joinpath(f"{name}/{dt.now().strftime("%d-%m-%Y_%H-%M")}")
    Path.mkdir(save_path)

    torch.save(
        model.state_dict(),
        save_path.joinpath("model"),
    )
    with open(save_path.joinpath("struct.txt"), "+w") as file:
        file.write(str(model))

    return save_path
