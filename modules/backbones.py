import torch
import torchvision


class MNISTCNN(torch.nn.Module):
    def __init__(self, **kwargs):
        super(MNISTCNN, self).__init__()
        self.cnn = torch.nn.Sequential(
            torch.nn.BatchNorm2d(1),
            torch.nn.Conv2d(1, 20, 5),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(2),
            #torch.nn.BatchNorm2d(20), 
            torch.nn.Conv2d(20, 50, 5),
            torch.nn.LeakyReLU(),
            torch.nn.MaxPool2d(2),
            #torch.nn.BatchNorm2d(50), 
        )

        self.output_shape = [-1, 800]

    def forward(self, x):
        out = self.cnn(x)
        return out.view(out.shape[0], -1)


if __name__ == "__main__":
    from datasets.loaders import get_mnist_split
    train_loader, valid_loader, test_loader = get_mnist_split(2)
    net = MNISTCNN()

    for X, y in train_loader:
        print("in: ", X.shape)
        print("out: ", net(X).shape)