import torch

class CNN(torch.nn.Module):
    def __init__(self, backbone, head):
        torch.nn.Module.__init__(self)

        self.backbone = backbone()
        self.head = head()

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

    def parameters(self):
        return list(self.backbone.parameters()) + list(self.head.parameters())
    
    def to(self, *args, **kwargs):
        self.backbone = self.backbone.to(*args, **kwargs)
        self.head = self.head.to(*args, **kwargs)
        return self




if __name__ == "__main__":
    from modules.backbones import MNISTCNN
    from modules.heads import MNISTFF
    from datasets.loaders import get_mnist_split

    loader,_,_,_ =get_mnist_split(5)
    net = CNN(MNISTCNN, MNISTFF)

    for X, y in loader:
        print("in: ", X.shape)
        print("out: ", net(X).shape)
        