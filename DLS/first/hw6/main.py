import os
from pathlib import Path

import numpy as np

from skimage.io import imread
from skimage.transform import resize
import torch
from torch.utils.data import DataLoader

# from models.segnet import SegNet
from models.unet import UNet
from lib.trainer import Trainer
from lib.losses import Losses
from lib.save_picture import save_picture


if __name__ == "__main__":
    THRESHOLD = 0.5
    BASE_PATH = Path(__file__).parent
    MAGIC_NUMER = 42
    SHUFFLE = False

    np.random.seed(MAGIC_NUMER)
    torch.manual_seed(MAGIC_NUMER)
    torch.cuda.manual_seed(MAGIC_NUMER)

    images = []
    lesions = []

    root = BASE_PATH.joinpath("PH2Dataset")
    for root, dirs, files in os.walk(os.path.join(root, "PH2 Dataset images")):
        if root.endswith("_Dermoscopic_Image"):
            images.append(imread(os.path.join(root, files[0])))
        if root.endswith("_lesion"):
            lesions.append(imread(os.path.join(root, files[0])))

    size = (256, 256)
    X = [resize(x, size, mode="constant", anti_aliasing=True) for x in images]
    Y = [resize(y, size, mode="constant", anti_aliasing=False) > 0.5 for y in lesions]

    ix = np.random.choice(len(X), len(X), False)
    tr, val, ts = np.split(ix, [100, 150])
    X = np.array(X, np.float32)
    Y = np.array(Y, np.float32)

    batch_size = 8
    train_dataloader = DataLoader(
        list(zip(np.rollaxis(X[tr], 3, 1), Y[tr, np.newaxis])),
        batch_size=batch_size,
        shuffle=SHUFFLE,
    )
    valid_dataloader = DataLoader(
        list(zip(np.rollaxis(X[val], 3, 1), Y[val, np.newaxis])),
        batch_size=batch_size,
        shuffle=SHUFFLE,
    )
    test_dataloader = DataLoader(
        list(zip(np.rollaxis(X[ts], 3, 1), Y[ts, np.newaxis])),
        batch_size=batch_size,
        shuffle=SHUFFLE,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet()
    loss_f = Losses.tversky_loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    schedule = torch.optim.lr_scheduler.StepLR(optimizer, 7, 0.1)
    trainer = Trainer(model, loss_f, optimizer, 25, device, save=True, threshold=THRESHOLD)

    history = trainer.train(train_dataloader, valid_dataloader)
    test_loss, test_iou = trainer.eval(train_dataloader)

    inputs, outputs = next(iter(test_dataloader))

    preds = trainer.model(inputs.to(device))
    preds = preds.detach().sigmoid().cpu()
    preds = preds > THRESHOLD
    title = f"Test Loss: {test_loss:.4}. Test IoU: {test_iou:.4}"
    save_picture(inputs, outputs, preds, path=trainer.saved_path.joinpath("test.png"), title=title)
