from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2

from models.discriminator import Discriminator
from models.generator import Generator

# from lib.dataset import DatasetPhoto
from lib.trainer import Trainer


if __name__ == "__main__":
    BASE_PATH = Path(__file__).parent
    MAGIC_NUMER = 42
    BATCH_SIZE = 16
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # np.random.seed(MAGIC_NUMER)
    # torch.manual_seed(MAGIC_NUMER)
    # torch.cuda.manual_seed(MAGIC_NUMER).

    # all_paths = list(BASE_PATH.joinpath("faces_dataset_small").iterdir())
    # train_dataset = DatasetPhoto(all_paths, "train")
    # train_dataloder = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=8)

    train_ds = ImageFolder(
        BASE_PATH.joinpath("photo"),
        transform=v2.Compose(
            [
                v2.Resize(Generator.IMG_SIZE),
                v2.CenterCrop(Generator.IMG_SIZE),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
    )

    # valid_dataloder = DataLoader(valid_dataset, BATCH_SIZE, shuffle=True, num_workers=4)
    train_dl = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=6, pin_memory=True)

    print("init model")
    EPOCHS = 100
    generator = Generator()
    discriminator = Discriminator()
    model = {
        "generator": generator,
        "discriminator": discriminator,
    }
    loss_f = {
        "generator": torch.nn.BCELoss(),
        "discriminator": torch.nn.BCELoss(),
    }
    optimizer = {
        "generator": torch.optim.Adam(generator.parameters(), lr=4e-4, betas=(0.5, 0.9)),
        "discriminator": torch.optim.Adam(discriminator.parameters(), lr=4e-4, betas=(0.5, 0.9)),
    }
    scheduler = {
        "generator": torch.optim.lr_scheduler.StepLR(optimizer["generator"], 10, 0.2),
        "discriminator": torch.optim.lr_scheduler.StepLR(optimizer["discriminator"], 10, 0.2),
    }
    trainer = Trainer(model, loss_f, optimizer, EPOCHS, DEVICE, scheduler=scheduler, save=True)
    history = trainer.train(train_dl)
    # trainer.check_profile(train_dataloder)

    # test_loss = trainer.eval(test_dataset)
    # img, _ = next(iter(dt_generator(test_dataset)))
    # pred, _ = model(img)

    # save_picture([pred], [img], BASE_PATH.joinpath("test.png"))
