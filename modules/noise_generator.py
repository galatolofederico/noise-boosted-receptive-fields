from datasets.utils import get_inf_generator
import numpy as np

class CyclicNoise:
    def __init__(self, loader):
        self.loader = loader
        self.len = len(loader)
        self.perm = np.random.permutation(self.len)
        self.i = -1

    def get(self):
        self.i = (self.i + 1) % self.len
        return self.loader[self.perm[self.i]][0]