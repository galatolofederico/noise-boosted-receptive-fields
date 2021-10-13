import os
import torch
import pytorch_lightning as pl
import sklearn.metrics
import argparse

from datasets.loaders import *

from modules.backbones import MNISTCNN
from modules.heads import MNISTRF, MNISTFF
from modules.noise_generator import CyclicNoise

from models.nbrf import NBRF
from models.cnn import CNN

class ClassifierModule(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        if self.hparams.dataset == "mnist":
            self.train_dataset, self.val_dataset, self.test_dataset, noise_loader = get_mnist_split(self.hparams.batch_size)
        elif self.hparams.dataset == "fmnist":
            self.train_dataset, self.val_dataset, self.test_dataset, noise_loader = get_fmnist_split(self.hparams.batch_size)
        elif self.hparams.dataset == "kmnist":
            self.train_dataset, self.val_dataset, self.test_dataset, noise_loader = get_kmnist_split(self.hparams.batch_size)
        else:
            raise Exception(f"Unknown dataset {self.hparams.dataset}")
        
        if self.hparams.model == "nbrf":
            self.model = NBRF(
                backbone = MNISTCNN,
                head = MNISTRF,
                n_rfs = 10,
                with_noise = self.hparams.with_noise,
                with_others = self.hparams.with_others,
                noise = self.hparams.noise,
                noise_generator = CyclicNoise(noise_loader),
                noise_position = self.hparams.noise_position,
            )
        elif self.hparams.model == "cnn":
            self.model = CNN(
                backbone = MNISTCNN,
                head = MNISTFF
            )
        else:
            raise Exception(f"Unknown model {self.hparams.model}")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_nb):
        x, y = batch
        out = self.model.forward(x)

        acc = (out.max(dim=1).indices == y).float().mean()
        loss = self.model.compute_loss(x, y)

        self.log('train_loss', loss)
        self.log('train_acc', acc)

        return loss

    def validation_step(self, batch, batch_nb):
        x, y = batch
        out = self.model.forward(x)

        acc = (out.max(dim=1).indices == y).float().mean()
        loss = self.model.compute_loss(x, y)

        self.log("val_loss", loss.item())
        self.log("val_acc", acc.item())

    def test_step(self, batch, batch_nb):
        x, y = batch
        out = self.model.forward(x)

        raw_preds = out.max(dim=1).indices
        preds5 = raw_preds.clone()
        preds9 = raw_preds.clone()

        preds5[out.max(dim=1).values < 0.5] = 11
        preds9[out.max(dim=1).values < 0.9] = 11

        results = {
            "targets": y,
            "raw_preds": raw_preds,
            "05_preds": preds5,
            "09_preds": preds9,
        }

        return results

    def test_epoch_end(self, outputs):
        y_05_pred = []
        y_09_pred = []
        y_raw_pred = []
        y_true = []
        for output in outputs:
            y_05_pred.extend(output["05_preds"].cpu().numpy())
            y_09_pred.extend(output["09_preds"].cpu().numpy())
            y_raw_pred.extend(output["raw_preds"].cpu().numpy())
            y_true.extend(output["targets"].cpu().numpy())

        ret_dict = {
            "test_raw_accuracy": sklearn.metrics.accuracy_score(y_true, y_raw_pred),
            "test_accuracy": sklearn.metrics.accuracy_score(y_true, y_09_pred),
            "test_09_accuracy": sklearn.metrics.accuracy_score(y_true, y_09_pred),
            "test_05_accuracy": sklearn.metrics.accuracy_score(y_true, y_05_pred),
        }

        self.results = ret_dict
        self.log_dict(ret_dict)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def train_dataloader(self):
        return self.train_dataset

    def val_dataloader(self):
        return self.val_dataset

    def test_dataloader(self):
        return self.test_dataset
