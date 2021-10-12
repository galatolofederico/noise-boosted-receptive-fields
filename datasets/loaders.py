import torchvision
import torch
from datasets.utils import get_split
from datasets.emnist10 import EMNIST10

noise_transforms = torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop(20),
    torchvision.transforms.Pad(4),
    torchvision.transforms.RandomAffine(10),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomPerspective(),
    torchvision.transforms.RandomRotation(180)
])

base_datasets_folder = "./datasets"

def get_dataset_split(dataset, loc, transforms, batch_size):
    args = dict(transform=transforms, download=True)
    train_loader, valid_loader, test_loader = get_split(dataset, batch_size, loc, **args)

    args["transform"] = torchvision.transforms.Compose([
        noise_transforms,
        transforms
    ])
    noise_loader = dataset(loc, **args, train=True)

    return train_loader, valid_loader, test_loader, noise_loader


def get_mnist_split(batch_size):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    loc = base_datasets_folder + "/mnist"

    return get_dataset_split(torchvision.datasets.MNIST, loc, transforms, batch_size)

def get_emnist_split(batch_size):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    loc = base_datasets_folder + "/emnist"

    return get_dataset_split(EMNIST10, loc, transforms, batch_size)

def get_kmnist_split(batch_size):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    loc = base_datasets_folder + "/kmnist"

    return get_dataset_split(torchvision.datasets.KMNIST, loc, transforms, batch_size)

def get_fmnist_split(batch_size):
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    loc = base_datasets_folder + "/fmnist"

    return get_dataset_split(torchvision.datasets.FashionMNIST, loc, transforms, batch_size)

if __name__ == "__main__":
    get_fmnist_split(10)
