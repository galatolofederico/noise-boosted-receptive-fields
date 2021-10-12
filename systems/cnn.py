import pytorch_lightning as pl
import torch

from systems.classifier import Classifier
from models.cnn import CNN

class CNNSystem(CNN, Classifier):
    def __init__(self, args):
        Classifier.__init__(self, args)
        CNN.__init__(self, args.backbone, args.head)

        self.save_hyperparameters(args)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def compute_loss(self, x, y):
        out = self.forward(x)
        return self.loss_fn(out, y)

