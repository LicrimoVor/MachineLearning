from typing import Literal, Callable
from inspect import isclass
from pathlib import Path
from time import sleep

import torch
from torch import optim
from torch.utils.data import Dataset
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt

from models.libs.save_model import save_model
from models.abstract import AbstractModule

from .dt_generator import dt_generator
from .save_picture import save_picture
from .create_path import create_path

plt.style.use("fivethirtyeight")
NOISE_FACTOR = 0.5


class Trainer:
    """Класс учителя моделей NN"""

    LOG_TEMPLATE = "Epoch {ep:03d}\nTrain_loss: {t_loss:0.6f} Val_loss {v_loss:0.6f}"
    SAVE_TEMPLATE = (
        "\nModels have {params} params\nLoss: {loss_f}\nOptimizer: {optimizer}"
        + "\nEpoch_count: {epoch_count}\nScheduler: {scheduler}"
    )
    BATCH_SIZE = 16

    def __init__(
        self,
        model: AbstractModule,
        loss_f: nn.modules.loss._WeightedLoss,
        optimizer: optim.Optimizer,
        epoch_count: int = 10,
        device: Literal["cpu", "cuda"] = "cpu",
        scheduler: optim.lr_scheduler.LRScheduler = None,
        scaler: bool = False,
        save: bool = True,
    ):
        self.device = torch.device(device)
        self.loss_f = loss_f
        if isclass(optimizer):
            assert Exception("Ожидался экземпляр класса Optimizer, а не сам класс")
        self.optimizer = optimizer
        self.epoch_count = epoch_count

        if isclass(scheduler):
            assert Warning("Scheduler не учитывается! Ожидался экземпляр класса, а не сам класс")
        self.scheduler = scheduler
        self.model = model.to(self.device)
        self.scaler = torch.amp.GradScaler() if scaler else None
        self.save = save
        self.save_path = create_path(self.model)

        self.train_loss = []
        self.val_loss = []

    def fit(self, train_loader: Dataset):
        """Одна эпоха обучения модели."""
        self.model.train()
        summ_loss = 0

        total = len(train_loader) // self.BATCH_SIZE

        for inputs, attrs in tqdm(
            dt_generator(train_loader, self.BATCH_SIZE), total=total, desc="Fit"
        ):
            inputs = inputs.to(self.device)
            noised_inputs = inputs + NOISE_FACTOR * torch.normal(0, 1, size=inputs.shape).to(
                self.device
            )
            self.optimizer.zero_grad()

            if self.scaler is None:
                outputs, _ = self.model(noised_inputs)
                loss = self.loss_f(outputs, inputs)
                loss.backward()
                self.optimizer.step()
            else:
                with torch.amp.autocast(str(self.device)):
                    outputs = self.model(noised_inputs)
                    loss = self.loss_f(outputs, inputs)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

            summ_loss += loss.detach().cpu().item()

        summ_loss /= total
        return summ_loss

    def eval(self, val_loader: Dataset):
        """Применение модели для валидации."""
        self.model.eval()
        summ_loss = 0
        processed_size = 0

        total = len(val_loader) // self.BATCH_SIZE

        for inputs, attrs in tqdm(
            dt_generator(val_loader, self.BATCH_SIZE), total=total, desc="Eval"
        ):
            inputs = inputs.to(self.device)
            noised_inputs = inputs + NOISE_FACTOR * torch.normal(0, 1, size=inputs.shape).to(
                self.device
            )

            with torch.no_grad():
                outputs, _ = self.model(noised_inputs)
                loss = self.loss_f(outputs, inputs)

            summ_loss += loss.item()
            processed_size += inputs.size(0)

        summ_loss /= total
        return summ_loss

    def train(
        self,
        train_loader: Dataset,
        val_loader: Dataset,
        callbacks: list[Callable[[int], None]] = [],
        step_save: int = 5,
    ):
        self.history = []
        predicts = []
        imgs = []

        try:
            for epoch in range(1, self.epoch_count + 1):
                print(f"Epoch: {epoch}")
                train_loss = self.fit(train_loader)

                if self.scheduler is not None:
                    self.scheduler.step()

                val_loss = self.eval(val_loader)
                self.history.append((train_loss, val_loss))

                print(
                    self.LOG_TEMPLATE.format(
                        ep=epoch,
                        t_loss=train_loss,
                        v_loss=val_loss,
                    ),
                    "\n" + "-" * 56,
                )
                for callback in callbacks:
                    callback(epoch)

                img, attr = next(iter(dt_generator(val_loader, 10)))
                with torch.no_grad():
                    outputs, _ = self.model(img.to(self.device))

                imgs.append(img)
                predicts.append(outputs)

                if epoch % step_save == 0:
                    save_picture(predicts, imgs, self.save_path.joinpath(f"imgs_{epoch}.png"))
                    imgs = []
                    predicts = []

        except KeyboardInterrupt:
            print("Предварительное окончание")
        finally:
            if not self.save:
                return self.history

            if isinstance(self.model, AbstractModule):
                self.model.save(self.save_path)
            else:
                save_model(self.model)

            with open(self.save_path.joinpath("record.txt"), "+w") as file:
                file.write(
                    self.LOG_TEMPLATE.format(
                        ep=epoch,
                        t_loss=train_loss,
                        v_loss=val_loss,
                    )
                )
                file.write(
                    self.SAVE_TEMPLATE.format(
                        params=self.model.get_count_params(),
                        loss_f=self.loss_f.__class__.__name__,
                        optimizer=self.optimizer.__repr__(),
                        epoch_count=self.epoch_count,
                        scheduler=self.scheduler.__class__.__name__,
                    )
                )

            self.save_history(self.history, self.save_path)
            print("SAVED")

        return self.history

    def save_history(self, history: list, path: Path):
        t_loss, val_loss = zip(*history)
        plt.clf()
        sleep(5)

        plt.plot(t_loss, label="train_loss")
        plt.plot(val_loss, label="val_loss")
        plt.legend(loc="best")
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.savefig(path.joinpath("record.png"))
