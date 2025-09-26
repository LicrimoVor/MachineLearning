from typing import Literal, Callable, TypedDict
from pathlib import Path
from time import sleep

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.profiler import profile, ProfilerActivity

from models.libs.save_model import save_model
from models.abstract import AbstractModule

from .save_picture import save_picture
from .create_path import create_path

plt.style.use("fivethirtyeight")
NOISE_FACTOR = 0.5


class GanDict[T](TypedDict):
    generator: T
    discriminator: T


class Trainer:
    """Класс учителя моделей GAN"""

    LOG_TEMPLATE = (
        "Epoch {ep:03d}\nGenerator: train - {t_loss_gen:0.6f};"
        + "\nDiscriminator: train - {t_loss_desc:0.6f};"
    )
    SAVE_TEMPLATE = "\nModels have {params} params" + "\nEpoch_count: {epoch_count}"

    def __init__(
        self,
        model: GanDict[AbstractModule],
        loss_f: GanDict[nn.modules.loss._WeightedLoss],
        optimizer: GanDict[optim.Optimizer],
        epoch_count: int = 10,
        device: Literal["cpu", "cuda"] = "cpu",
        scheduler: GanDict[optim.lr_scheduler.LRScheduler] = None,
        scaler: bool = False,
        save: bool = True,
    ):
        self.device = torch.device(device)
        self.loss_f = loss_f
        self.optimizer = optimizer
        self.epoch_count = epoch_count
        self.scheduler = scheduler

        self.discriminator = model["discriminator"].to(self.device)
        self.generator = model["generator"].to(self.device)
        self.scaler = torch.amp.GradScaler() if scaler else None
        self.save = save
        self.save_path = create_path(name="GAN")

        self.train_loss = []
        self.val_loss = []

    def fit(self, train_loader: DataLoader):
        """Одна эпоха обучения модели."""
        self.generator.train()
        self.discriminator.train()
        loss_discriminator = []
        loss_generator = []

        for photos, _ in tqdm(train_loader):
            photos = photos.to(self.device)
            labels_real = torch.ones((photos.size(0), 1), device=self.device)
            labels_fake = torch.zeros((photos.size(0), 1), device=self.device)

            # discriminator
            self.optimizer["discriminator"].zero_grad()
            preds = self.discriminator(photos)
            loss_real = self.loss_f["discriminator"](preds, labels_real)

            latent = self.generator.gen_latent(photos.size(0)).to(self.device)
            fake_photos = self.generator(latent).detach()
            preds = self.discriminator(fake_photos)
            loss_fake = self.loss_f["discriminator"](preds, labels_fake)

            loss = loss_fake + loss_real
            loss.backward()
            self.optimizer["discriminator"].step()
            loss_discriminator.append(loss.cpu().item())

            # generator
            self.optimizer["generator"].zero_grad()
            latent = self.generator.gen_latent(photos.size(0)).to(self.device)
            fake_photos = self.generator(latent)
            preds = self.discriminator(fake_photos)
            loss_fake = self.loss_f["generator"](preds, labels_real)

            loss_fake.backward()
            self.optimizer["generator"].step()
            loss_generator.append(loss_fake.cpu().item())

        return np.mean(loss_generator), np.mean(loss_discriminator)

    def train(
        self,
        train_loader: DataLoader,
        callbacks: list[Callable[[int], None]] = [],
        step_save: int = 5,
    ):
        self.history = {"generator": [], "discriminator": []}
        predicts = []
        imgs = []

        try:
            for epoch in range(1, self.epoch_count + 1):
                print(f"Epoch: {epoch}")
                t_loss_gen, t_loss_desc = self.fit(train_loader)

                if self.scheduler is not None:
                    self.scheduler["discriminator"].step()
                    self.scheduler["generator"].step()

                self.history["generator"].append(t_loss_gen)
                self.history["discriminator"].append(t_loss_desc)

                print(
                    self.LOG_TEMPLATE.format(
                        ep=epoch,
                        t_loss_gen=t_loss_gen,
                        t_loss_desc=t_loss_desc,
                    ),
                    "\n" + "-" * 56,
                )
                for callback in callbacks:
                    callback(epoch)

                latent = self.generator.gen_latent(10).to(self.device)
                with torch.no_grad():
                    fake_photos = self.generator(latent)
                    preds = self.discriminator(fake_photos)

                imgs.append(fake_photos)
                predicts.append(preds)

                if epoch % step_save == 0:
                    save_picture(imgs, predicts, self.save_path.joinpath(f"imgs_{epoch}.png"))
                    imgs = []
                    predicts = []
        except KeyboardInterrupt:
            print("Предварительное окончание")
        finally:
            if not self.save:
                return self.history

            save_model(self.discriminator, self.save_path)
            save_model(self.generator, self.save_path)

            # with open(self.save_path.joinpath("record.txt"), "+w") as file:
            #     file.write(
            #         self.LOG_TEMPLATE.format(
            #             ep=epoch,
            #             t_loss_gen=t_loss_gen,
            #             v_loss_gen=v_loss_gen,
            #             t_loss_desc=t_loss_desc,
            #             v_loss_desc=v_loss_desc,
            #         ),
            #     )
            #     file.write(
            #         self.SAVE_TEMPLATE.format(
            #             params=self.model.get_count_params(),
            #             epoch_count=self.epoch_count,
            #         )
            #     )

            self.save_history(self.history, self.save_path)
            print("SAVED")

        return self.history

    def save_history(self, history: dict[str, list], path: Path):
        plt.clf()
        sleep(5)

        plt.plot(history["generator"], label="train loss generator")
        plt.plot(history["discriminator"], label="train loss discriminator")

        plt.legend(loc="best")
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.savefig(path.joinpath("record.png"))

    def check_profile(self, dataloader: DataLoader):

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            profile_memory=True,
            record_shapes=True,
            # with_stack=True,
        ) as prof:
            try:
                self.fit(dataloader)
            except KeyboardInterrupt:
                print("end")
                print(
                    prof.key_averages(group_by_stack_n=5).table(
                        sort_by="self_cpu_time_total", row_limit=20
                    )
                )
                prof.key_averages(group_by_input_shape=True).table(
                    sort_by="cpu_time_total", row_limit=10
                )

        prof.export_chrome_trace("trace.json")
