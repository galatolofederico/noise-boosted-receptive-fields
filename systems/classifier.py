import os
import torch
import pytorch_lightning as pl
import sklearn
from sklearn.metrics import confusion_matrix
import csv
import argparse
class Classifier(pl.LightningModule):
    def __init__(self, args):
        pl.LightningModule.__init__(self)
        self.args = args

        self.train_dataset = args.train_dataset
        self.val_dataset = args.valid_dataset
        self.test_dataset = args.test_dataset

    def training_step(self, batch, batch_nb):
        x, y = batch
        out = self.forward(x)

        acc = (out.max(dim=1).indices == y).float().mean()

        loss = self.compute_loss(x, y)

        self.log('train_loss', loss)
        self.log('train_acc', acc)

        return loss

    def validation_step(self, batch, batch_nb):
        x, y = batch
        out = self.forward(x)

        acc = (out.max(dim=1).indices == y).float().mean()
        loss = self.compute_loss(x, y)

        self.log("val_loss", loss.item())
        self.log("val_acc", acc.item())

    def test_step(self, batch, batch_nb):
        x, y = batch
        out = self.forward(x)

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
        return torch.optim.Adam(self.parameters(), lr=self.args.lr)

    def manual_log(self, file):
        exists = os.path.exists(file)
        resultsfile = open(file, "a")

        results = {
            "model": self.args.model,
            "trained_with": self.args.dataset,
            "without_noise": self.args.without_noise,
            "without_other": self.args.without_other,
            "noise": self.args.noise,
            "noise_position": self.args.noise_position,
            "accuracy": self.results["test_accuracy"],
            "05_accuracy": self.results["test_05_accuracy"],
            "09_accuracy": self.results["test_09_accuracy"],
            "raw_accuracy": self.results["test_raw_accuracy"]
        }

        writer = csv.DictWriter(resultsfile, fieldnames=results.keys())
        if not exists: writer.writeheader()
        writer.writerow(results)
        resultsfile.close()

    def train_dataloader(self):
        return self.train_dataset

    def val_dataloader(self):
        return self.val_dataset

    def test_dataloader(self):
        return self.test_dataset
