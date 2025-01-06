from datetime import datetime as dt
from pathlib import Path

from models.abstract import AbstractModule

BASE_DIR = Path(__file__).parent.parent.joinpath("models")


def create_path(model: AbstractModule = None, name: str = None):
    if model is None and name is None:
        return None
    name = name if model is None else model.name

    if not BASE_DIR.joinpath(f"{name}").exists():
        BASE_DIR.joinpath(f"{name}").mkdir()

    save_path = BASE_DIR.joinpath(f"./{name}/{dt.now().strftime("%d-%m-%Y_%H-%M")}")
    Path.mkdir(save_path)

    return save_path
