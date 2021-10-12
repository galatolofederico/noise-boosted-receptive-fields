import torchvision
import torch
import os
import pickle

class EMNIST10(torch.utils.data.Dataset):
    def __init__(self, loc, transform, download, train=True,):
        self.cache = "datasets/emnist_cache_" + str(train) + ".pkl"
        self.dataset = []
        self.chars10 = [0, 1, 2, 3, 4, 5, 6, 7, 10, 12]  # abcdefghkm

        # Builds the dataset from the EMNIST one, taking only the relevant letters
        if not os.path.exists(self.cache):
            emnist = torchvision.datasets.EMNIST(loc, download=True, split="letters", train=train, transform=transform)

            for x, y in emnist:
                if y in self.chars10:
                    self.dataset.append([x, y])
            pickle.dump(self.dataset, open(self.cache, "wb"))
        else:
            self.dataset = pickle.load(open(self.cache, "rb"))


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        return self.dataset[i]


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    import numpy as np

    # Checks emnist10 samples

    ds = EMNIST10("tmp/emnist", download=True, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]), train=True)
    for x, y in ds:
        print(chr(y + ord('a') - 1))
        plt.imshow(np.transpose(x[0]))
        plt.show()
