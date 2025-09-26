from datetime import datetime as dt
from pathlib import Path

from torch import nn
import torch

BASE_DIR = Path(__file__).parent


class AbstractModule(nn.Module):
    name: str

    def save(self):
        """Сохраняет модель и возвращает путь к папке с файлами."""
        if not BASE_DIR.joinpath(f"{self.name}").exists():
            BASE_DIR.joinpath(f"{self.name}").mkdir()

        save_path = BASE_DIR.joinpath(f"./{self.name}/{dt.now().strftime("%d-%m-%Y_%H-%M")}")
        Path.mkdir(save_path)

        torch.save(
            self.state_dict(),
            save_path.joinpath("model"),
        )
        with open(save_path.joinpath("struct.txt"), "+w") as file:
            file.write(str(self))

        return save_path

    @classmethod
    def load(cls, n_classes: int, path: str):
        model = cls(n_classes)
        model.load_state_dict(torch.load(path, weights_only=True))
        model.eval()
        return model
