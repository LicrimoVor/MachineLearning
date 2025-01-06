from pathlib import Path

from sklearn.model_selection import train_test_split
import torch

from models.AE import Autoencoder
from lib.dataset import DatasetPhoto
from lib.fetch_dataset import fetch_dataset
from lib.trainer import Trainer
from lib.dt_generator import dt_generator
from lib.save_picture import save_picture


if __name__ == "__main__":
    THRESHOLD = 0.5
    BASE_PATH = Path(__file__).parent.joinpath("data")
    MAGIC_NUMER = 42
    BATCH_SIZE = 8
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # np.random.seed(MAGIC_NUMER)
    # torch.manual_seed(MAGIC_NUMER)
    # torch.cuda.manual_seed(MAGIC_NUMER).

    print("load data")
    data, attrs = fetch_dataset(
        BASE_PATH.joinpath("lfw_attributes.txt"), BASE_PATH.joinpath("lfw-deepfunneled")
    )

    X_train, X_test, y_train, y_test = train_test_split(data, attrs, test_size=0.3, shuffle=True)

    train_dataset = DatasetPhoto(X_train, y_train, "train")
    # valid_dataset = DatasetPhoto(X_valid, y_valid, "valid")
    test_dataset = DatasetPhoto(X_test, y_test, "test")

    print("init model")
    DIM_CODE = 128
    EPOCHS = 40
    model = Autoencoder(DIM_CODE, "conv")
    # model = Autoencoder.load(
    #     DIM_CODE, "conv", BASE_PATH.joinpath("models/AE_conv/30-11-2024_21-16/model")
    # )

    loss_f = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    schedule = torch.optim.lr_scheduler.StepLR(optimizer, 12, 0.1)
    trainer = Trainer(model, loss_f, optimizer, EPOCHS, DEVICE, save=True)

    history = trainer.train(train_dataset, test_dataset)
    test_loss = trainer.eval(test_dataset)
    img, _ = next(iter(dt_generator(test_dataset)))
    pred, _ = model(img)

    save_picture([pred], [img], BASE_PATH.joinpath("test.png"))
