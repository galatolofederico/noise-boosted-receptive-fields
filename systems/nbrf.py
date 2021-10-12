import pytorch_lightning as pl
import torch
import random

from systems.classifier import Classifier
from models.nbrf import NBRF
from modules.backbones import MNISTCNN
from modules.heads import MNISTRF

class NBRFSystem(NBRF, Classifier):
    def __init__(self, args):
        Classifier.__init__(self, args)
        NBRF.__init__(self, args.backbone, args.head, args.n_rfs, args.with_noise, args.with_other)
        
        self.noise_generator = args.noise_generator
        self.noise = args.noise
        self.noise_position = args.noise_position

    def compute_loss(self, x, y, noise_perc = 1.0, noise_elems = 3):
        noise_dim = int(x.shape[0]*noise_perc)
        if self.with_noise:
            if self.noise == "white":
                noise_channels = torch.randn(noise_dim, noise_elems, *x.shape[1:]).to(x.device) * x.std() + x.mean()
            if self.noise == "dataset":
                noise_channels = torch.zeros(noise_dim, noise_elems, *x.shape[1:]).to(x.device)
                for i in range(0, noise_dim):
                    for j in range(0, noise_elems):
                        noise_channels[i,j] = self.noise_generator.get()

            if self.noise == "batch":
                noise_channels = torch.zeros(noise_dim, noise_elems, *x.shape[1:]).to(x.device)
                for i in range(0, noise_dim):
                    classes = []
                    for j in range(0, noise_elems):
                        elem = random.randint(0, x.shape[0] - 1)
                        while y[elem] in classes: elem = random.randint(0, x.shape[0] - 1)
                        noise_channels[i,j,:] = x[elem]

            x_features = self.compute_features(x)

            if self.noise_position == "head":
                noise_batch = noise_channels.view(-1, *x.shape[1:])
                noise_features = self.compute_features(noise_batch).view(noise_dim, noise_elems, x_features.shape[1]).mean(dim=1)

            if self.noise_position == "backbone":
                noise_features = self.compute_features(noise_channels.mean(dim=1))

            outs = self.compute_rf_outputs(torch.cat((noise_features, x_features), dim=0))
        else:
            x_features = self.compute_features(x)
            outs = self.compute_rf_outputs(x_features)

        count = torch.bincount(y,minlength=self.args.n_outputs)
        losses = torch.zeros(len(self.rfs)).to(x.device)

        for i, rf in enumerate(self.rfs):
            others_mask = torch.arange(0, len(self.rfs)) != i

            current_count = count[i].float()
            others_count = count[others_mask].sum().float()
            noise_count = noise_dim

            weights = None
            if self.with_noise and not self.with_others: weights = current_count / torch.tensor([noise_count, current_count], device=current_count.device)
            elif self.with_others and not self.with_noise: weights = current_count / torch.tensor([others_count, current_count], device=current_count.device)
            elif self.with_noise and self.with_others: weights = current_count / torch.tensor([noise_count, others_count, current_count], device=current_count.device)

            class_lables = torch.zeros(x.shape[0])
            if self.with_others and self.with_noise:
                class_lables[y != i] = torch.ones(x.shape[0])[y != i]
                class_lables[y == i] = 2*torch.ones(x.shape[0])[y == i]
            else:
                class_lables[y == i] = torch.ones(x.shape[0])[y == i]

            labels = None
            if self.with_noise: labels = torch.cat((torch.zeros(noise_dim), class_lables)).long().to(x.device)
            elif self.with_others: labels = class_lables.long().to(x.device)

            loss_fn = torch.nn.CrossEntropyLoss(weights).to(x.device)
            losses[i] = loss_fn(outs[i], labels)
        return losses.sum()
