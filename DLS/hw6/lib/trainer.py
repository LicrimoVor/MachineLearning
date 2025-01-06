from typing import Literal, Callable
from inspect import isclass
from pathlib import Path

import torch
from torch import optim
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
from torchmetrics import JaccardIndex
import matplotlib.pyplot as plt

from models.libs.save_model import save_model
from models.abstract import AbstractModule

plt.style.use("fivethirtyeight")


class Trainer:
    """Класс учителя моделей NN"""

    LOG_TEMPLATE = (
        "Epoch {ep:03d}\nTrain_loss: {t_loss:0.4f} "
        + "Val_loss {v_loss:0.4f}\nTrain_iou {t_iou:0.4f} "
        + "Val_iou {v_iou:0.4f}"
    )
    SAVE_TEMPLATE = (
        "\nLoss: {loss_f}\nOptimizer: {optimizer}"
        + "\nEpoch_count: {epoch_count}\nScheduler: {scheduler}"
    )

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
        threshold: float = 0.5,
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
        self.iou_score = JaccardIndex(threshold=threshold, task="binary", average="none").to(device)

        self.train_loss = []
        self.val_loss = []

    def predict(self, test_loader: DataLoader):
        """Ответ модели."""
        with torch.no_grad():
            logits = []

            for inputs in test_loader:
                inputs = inputs.to(self.device)
                self.model.eval()
                outputs = self.model(inputs).cpu()
                logits.append(outputs)

        probs = nn.functional.softmax(torch.cat(logits), dim=-1).numpy()
        return probs

    def fit(self, train_loader: DataLoader):
        """Одна эпоха обучения модели."""
        self.model.train()
        summ_loss = 0
        summ_iou = 0

        for inputs, labels in tqdm(train_loader, desc="Fit"):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()

            if self.scaler is None:
                outputs = self.model(inputs)
                loss = self.loss_f(outputs, labels)
                loss.backward()
                self.optimizer.step()
            else:
                with torch.amp.autocast(str(self.device)):
                    outputs = self.model(inputs)
                    loss = self.loss_f(outputs, labels)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

            summ_loss += loss.detach().cpu().item()
            summ_iou += self.iou_score(outputs, labels).detach().cpu().item()

        summ_loss /= len(train_loader)
        summ_iou /= len(train_loader)
        return summ_loss, summ_iou

    def eval(self, val_loader: DataLoader):
        """Применение модели для валидации."""
        self.model.eval()
        summ_loss = 0
        summ_iou = 0
        processed_size = 0

        for inputs, labels in tqdm(val_loader, desc="Eval"):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            with torch.set_grad_enabled(False):
                outputs = self.model(inputs)
                loss = self.loss_f(outputs, labels)

            summ_loss += loss.item()
            summ_iou += self.iou_score(outputs, labels).cpu().item()
            processed_size += inputs.size(0)

        summ_loss /= len(val_loader)
        summ_iou /= len(val_loader)
        return summ_loss, summ_iou

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        callbacks: list[Callable[[int], None]] = [],
    ):
        self.history = []

        try:
            for epoch in range(1, self.epoch_count + 1):
                print(f"Epoch: {epoch}")
                train_loss, train_iou = self.fit(train_loader)

                if self.scheduler is not None:
                    self.scheduler.step()

                val_loss, val_iou = self.eval(val_loader)
                self.history.append((train_loss, train_iou, val_loss, val_iou))

                print(
                    self.LOG_TEMPLATE.format(
                        ep=epoch,
                        t_loss=train_loss,
                        v_loss=val_loss,
                        t_iou=train_iou,
                        v_iou=val_iou,
                    ),
                    "\n" + "-" * 56,
                )
                for callback in callbacks:
                    callback(epoch)

        except KeyboardInterrupt:
            print("Предварительное окончание")
        except Exception as ex:
            print(ex)
        finally:
            if not self.save:
                return self.history

            if isinstance(self.model, AbstractModule):
                save_path = self.model.save()
            else:
                save_path = save_model(self.model)

            with open(save_path.joinpath("record.txt"), "+w") as file:
                file.write(
                    self.LOG_TEMPLATE.format(
                        ep=epoch,
                        t_loss=train_loss,
                        v_loss=val_loss,
                        t_iou=train_iou,
                        v_iou=val_iou,
                    )
                )
                file.write(
                    self.SAVE_TEMPLATE.format(
                        loss_f=self.loss_f.__class__.__name__,
                        optimizer=self.optimizer.__repr__(),
                        epoch_count=self.epoch_count,
                        scheduler=self.scheduler.__class__.__name__,
                    )
                )

            self.save_history(self.history, save_path)
            self.saved_path = save_path
            print("SAVED")

        return self.history

    def save_history(self, history: list, path: Path):
        t_loss, t_iou, val_loss, val_iou = zip(*history)
        fig, axs = plt.subplots(2, 1)

        axs[0].plot(t_iou, label="iou")
        axs[0].plot(val_iou, label="val_iou")
        axs[0].legend(loc="best")
        axs[0].set_ylabel("iou")

        axs[1].plot(t_loss, label="train_loss")
        axs[1].plot(val_loss, label="val_loss")
        axs[1].legend(loc="best")
        axs[1].set_xlabel("epochs")
        axs[1].set_ylabel("loss")
        fig.savefig(path.joinpath("record.png"))
