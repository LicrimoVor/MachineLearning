from typing import Literal, Callable
from inspect import isclass

import torch
from torch import optim
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt

from models.libs.save_model import save_model
from models.abstract import AbstractModule


class Trainer:
    """Класс учителя моделей NN"""

    LOG_TEMPLATE = (
        "Epoch {ep:03d} train_loss: {t_loss:0.4f} "
        + "val_loss {v_loss:0.4f} train_acc {t_acc:0.4f} "
        + "val_acc {v_acc:0.4f}"
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
    ):
        self.device = torch.device(device)
        self.loss_f = loss_f.to(self.device)
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
        running_loss = 0.0
        running_corrects = 0
        processed_data = 0

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

            preds = torch.argmax(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            processed_data += inputs.size(0)

        train_loss = running_loss / processed_data
        train_acc = running_corrects.cpu().numpy() / processed_data
        return train_loss, train_acc

    def eval(self, val_loader: DataLoader):
        """Применение модели для валидации."""
        self.model.eval()
        running_loss = 0.0
        running_corrects = 0
        processed_size = 0

        for inputs, labels in tqdm(val_loader, desc="Eval"):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            with torch.set_grad_enabled(False):
                outputs = self.model(inputs)
                loss = self.loss_f(outputs, labels)
                preds = torch.argmax(outputs, 1)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            processed_size += inputs.size(0)

        val_loss = running_loss / processed_size
        val_acc = running_corrects.double() / processed_size
        return val_loss, val_acc.cpu().item()

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
                train_loss, train_acc = self.fit(train_loader)

                if self.scheduler is not None:
                    self.scheduler.step()

                print("loss", train_loss)

                val_loss, val_acc = self.eval(val_loader)
                self.history.append((train_loss, train_acc, val_loss, val_acc))

                print(
                    self.LOG_TEMPLATE.format(
                        ep=epoch,
                        t_loss=train_loss,
                        v_loss=val_loss,
                        t_acc=train_acc,
                        v_acc=val_acc,
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
                        t_acc=train_acc,
                        v_acc=val_acc,
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

                loss, acc, val_loss, val_acc = zip(*self.history)
                fig, axs = plt.subplots(2, 1)

                axs[0].plot(acc, label="acc")
                axs[0].plot(val_acc, label="val_acc")
                axs[0].legend(loc="best")
                axs[0].set_xlabel("epochs")
                axs[0].set_ylabel("acc")

                axs[1].plot(loss, label="train_loss")
                axs[1].plot(val_loss, label="val_loss")
                axs[1].legend(loc="best")
                axs[1].set_xlabel("epochs")
                axs[1].set_ylabel("loss")
                fig.savefig(save_path.joinpath("record.png"))
                print("SAVED")

        return self.history
