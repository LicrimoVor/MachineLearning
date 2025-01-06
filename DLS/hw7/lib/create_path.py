from datetime import datetime as dt
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent.joinpath("models")


def create_path(model):
    if not BASE_DIR.joinpath(f"{model.name}").exists():
        BASE_DIR.joinpath(f"{model.name}").mkdir()

    save_path = BASE_DIR.joinpath(f"./{model.name}/{dt.now().strftime("%d-%m-%Y_%H-%M")}")
    Path.mkdir(save_path)

    return save_path
