import os

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import BinaryPrecision, BinaryRecall
from torchvision import transforms as T
from tqdm import tqdm

from settings.config import Config


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_transforms():
    transforms = T.Compose(
        [
            T.Resize(size=Config.resize_to, antialias=None),
            T.Normalize(mean=Config.mean, std=Config.std),
        ]
    )
    return transforms


class Trainer:
    def __init__(
        self,
        model,
        criterion,
        optimizer,
        train_dataloader,
        val_dataloader,
        epochs,
        device,
        lr_scheduler=None,
        log_dir=Config.log_dir,
        min_val_loss=np.inf,
        model_name="model",
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.epochs = epochs
        self.device = device
        self.lr_scheduler = lr_scheduler
        self.writer = SummaryWriter(log_dir=os.path.join(log_dir, model_name))
        self.min_val_loss = min_val_loss
        self.model_name = model_name

    def train_step(self):
        self.model.train()
        batch_loss = 0
        for batch in tqdm(self.train_dataloader, leave=False):
            imgs1, imgs2, label = batch
            imgs1 = imgs1.to(self.device)
            imgs2 = imgs2.to(self.device)
            label = label.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(imgs1, imgs2)
            loss = self.criterion(output, label)
            loss.backward()
            self.optimizer.step()

            batch_loss += loss.item()

        return batch_loss / len(self.train_dataloader)

    def val_step(self):
        self.model.eval()
        precision = BinaryPrecision().to(self.device)
        recall = BinaryRecall().to(self.device)
        batch_loss = 0
        batch_precision = 0
        batch_recall = 0
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, leave=False):
                imgs1, imgs2, label = batch
                imgs1 = imgs1.to(self.device)
                imgs2 = imgs2.to(self.device)
                label = label.to(self.device)

                output = self.model(imgs1, imgs2)
                loss = self.criterion(output, label)

                batch_loss += loss.item()
                batch_precision += precision(output, label)
                batch_recall += recall(output, label)

        batch_loss /= len(self.val_dataloader)
        batch_precision /= len(self.val_dataloader)
        batch_recall /= len(self.val_dataloader)

        if batch_loss < self.min_val_loss:
            # remove previous models with the same name
            for model in os.listdir(Config.model_path):
                if self.model_name in model:
                    os.remove(os.path.join(Config.model_path, model))

            self.min_val_loss = batch_loss
            torch.save(
                self.model.state_dict(),
                os.path.join(Config.model_path, self.model_name + "_" + f"{str(batch_loss).replace('.', '_')[:7]}.pth"),
            )
            print("Best model saved!")

        return batch_loss, batch_precision, batch_recall

    def fit(self):
        self.model.to(self.device)
        for epoch in range(self.epochs):
            train_loss = self.train_step()
            val_loss, val_precision, val_recall = self.val_step()
            if self.lr_scheduler:
                self.lr_scheduler.step(val_loss)

            self.writer.add_scalar("Loss/train", train_loss, epoch)
            self.writer.add_scalar("Loss/valid", val_loss, epoch)
            self.writer.add_scalar("Precision/valid", val_precision, epoch)
            self.writer.add_scalar("Recall/valid", val_recall, epoch)
            self.writer.add_scalar("LR", self.optimizer.param_groups[0]["lr"], epoch)
            self.writer.close()

            print(
                f"Epoch {epoch+1}/{self.epochs} | Train loss: {train_loss:.5f} | Val loss: {val_loss:.5f} | Val precision: "
                f"{val_precision:.5f} | Val recall: {val_recall:.5f}"
            )
