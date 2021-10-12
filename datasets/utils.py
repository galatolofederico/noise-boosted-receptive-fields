import torch
import torchvision
import random


def get_split(dataset, batch_size, *largs, **kwargs):
        valid_size = 0.1

        test_dataset = dataset(*largs, **kwargs)
        train_dataset = dataset(*largs, **kwargs, train=True)
        valid_dataset = dataset(*largs, **kwargs, train=True)

        num_train = len(train_dataset)
        indices = list(range(num_train))
        split = int(valid_size * num_train)

        random.shuffle(indices)
        train_idx, valid_idx = indices[split:], indices[:split]

        train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
        valid_sampler = torch.utils.data.SubsetRandomSampler(valid_idx)

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, sampler=train_sampler
        )

        valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=batch_size, sampler=valid_sampler
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=batch_size
        )

        return train_loader, valid_loader, test_loader


def get_inf_generator(dataset):
    while True:
        for elem in dataset:
            yield elem
