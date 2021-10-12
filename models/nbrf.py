import torch

class NBRF(torch.nn.Module):
    def __init__(self, backbone, head, n_rfs, with_noise, with_others):
        torch.nn.Module.__init__(self)

        self.with_noise = with_noise
        self.with_others = with_others
        self.backbone = backbone()

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



if __name__ == "__main__":
    from modules.backbones import MNISTCNN
    from modules.heads import MNISTRF
    from datasets.loaders import get_mnist_loader

    loader = get_mnist_loader(train=True, batch_size=2)
    net = NBRF(MNISTCNN, MNISTRF, 10, True, True)

    for X, y in loader:
        print("in: ", X.shape)
        print("out: ", net(X).shape)
