import torch
import torchvision

class MNISTRF(torch.nn.Module):
    def __init__(self, n_outputs, hiddens=500, **kwargs):
        super(MNISTRF, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(800, hiddens),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hiddens, n_outputs),
            torch.nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        return self.net(x)

class MNISTFF(torch.nn.Module):
    def __init__(self, n_outputs=10, hiddens=5000, **kwargs):
        super(MNISTFF, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(800, hiddens),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hiddens, n_outputs),
            torch.nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        return self.net(x)
