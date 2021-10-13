import torch
import random

class NBRF(torch.nn.Module):
    def __init__(self, backbone, head, n_rfs, with_noise, with_others, noise, noise_position, noise_generator):
        torch.nn.Module.__init__(self)

        self.with_noise = with_noise
        self.with_others = with_others
        self.noise = noise
        self.noise_position = noise_position
        self.noise_generator = noise_generator

        self.backbone = backbone()
        self.n_outputs = n_rfs

        self.rfs = torch.nn.ModuleList()
        for i in range(n_rfs):
            self.rfs.append(head(3 if self.with_noise and self.with_others else 2))

    def parameters(self):
        ret = []
        ret += list(self.backbone.parameters())
        for rf in self.rfs:
            ret += list(rf.parameters())
        return ret

    def to(self, *args, **kwargs):
        self.backbone = self.backbone.to(*args, **kwargs)
        for i in range(0, len(self.rfs)):
            self.rfs[i] = self.rfs[i].to(*args, **kwargs)
        return self

    def compute_features(self, x):
        return self.backbone(x)


    def compute_rf_outputs(self, features):
        ret = []
        for rf in self.rfs:
            ret.append(rf(features))
        return ret

    def compute(self, x):
        features = self.compute_features(x)
        return self.compute_rf_outputs(features)

    def forward(self, x):
        outs = self.compute(x)
        votes = torch.zeros(x.shape[0], len(self.rfs), device=x.device)
        for i, out in enumerate(outs):
            if self.with_others and self.with_noise:
                rf_out = (1 - out[:,0]) * (1 - out[:,1]) * (out[:,2])
            else:
                rf_out = (1 - out[:,0]) * (out[:,1])

            votes[:, i] = rf_out
        return votes


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

        count = torch.bincount(y,minlength=self.n_outputs)
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



if __name__ == "__main__":
    from modules.backbones import MNISTCNN
    from modules.heads import MNISTRF
    from datasets.loaders import get_mnist_loader

    loader = get_mnist_loader(train=True, batch_size=2)
    net = NBRF(MNISTCNN, MNISTRF, 10, True, True)

    for X, y in loader:
        print("in: ", X.shape)
        print("out: ", net(X).shape)
