from pathlib import Path

DATA_DIR = Path(__file__).parent.joinpath("data")
TRAIN_DIR = DATA_DIR.joinpath("train/")
TEST_DIR = DATA_DIR.joinpath("testset/")
RESCALE_SIZE = 256
DEVICE = "cuda"
