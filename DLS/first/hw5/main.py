import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader

from settings import TRAIN_DIR, TEST_DIR

from models.meduim import MediumCnn

# from models.simple import SimpleCnn
from models.abstract import BASE_DIR

# from models.resnet import create_resnet34, load_resnet34
from libs.dataset import SimpsonsDataset
from libs.trainer import Trainer


if __name__ == "__main__":

    train_val_files = sorted(list(TRAIN_DIR.rglob("*.jpg")))
    test_files = sorted(list(TEST_DIR.rglob("*.jpg")))

    train_val_labels = [path.parent.name for path in train_val_files]
    train_files, val_files = train_test_split(
        train_val_files, test_size=0.25, stratify=train_val_labels
    )
    val_dataset = SimpsonsDataset(val_files, mode="val")
    train_dataset = SimpsonsDataset(train_files, mode="train")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    n_classes = len(np.unique(train_val_labels))
    # model = create_resnet34(n_classes)
    # model = load_resnet34(n_classes, BASE_DIR.joinpath("ResNet/09-11-2024_20-05/model"))
    # model = MediumCnn(n_classes)
    model = MediumCnn.load(n_classes, BASE_DIR.joinpath("Medium/10-11-2024_11-11/model"))

    # def callback(epoch: int):
    #     if epoch != 4:
    #         return None

    #     for param in model.parameters():
    #         param.requires_grad = True

    loss_f = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    schedule = torch.optim.lr_scheduler.StepLR(optimizer, 2, 0.1)
    trainer = Trainer(model, loss_f, optimizer, 5, "cuda", save=True)

    history = trainer.train(train_loader, val_loader)
