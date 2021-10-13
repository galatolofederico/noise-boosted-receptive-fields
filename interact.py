import torch
import sys, os
import numpy as np
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import argparse
from datasets.loaders import *
from classifier import ClassifierModule


parser = argparse.ArgumentParser(description='Test a model against a dataset')

parser.add_argument('--dataset',
                    choices=["mnist","fmnist", "kmnist"],
                    help='Dataset to use',
                    default=None,
                    required=True
                    )

parser.add_argument('--model',
                    help='A model checkpoint',
                    type=str,
                    required=True
                    )

parser.add_argument('--th',
                    type=float,
                    help='Out of class threshold',
                    default=0.5
                    )

parser.add_argument('--onlyifactive',
                    const=True,
                    default=False,
                    action='store_const',
                    help='Plot only if at least 1 receptive field is active'
                    )

def plot(net, dataset, device, args):
    for x, y in dataset:
        c_out = net(x.to(device))[0].detach().cpu()
        if hasattr(net.model, "compute"): #nbrf models
            outs = net.model.compute(x.to(device))
        else: #cnn models
            outs = [torch.tensor([[o]]) for o in c_out]

        c_value = c_out.max(0)[1].item()
        if args.onlyifactive and (c_out > args.th).sum() == 0:
            continue

        f1, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        ax1.imshow(x[0,0])
        ax2.axhline(y=args.th, color='r', linestyle='--')
        w = 1
        if outs is not None:
            for i, out in enumerate(outs):
                colors = ""
                if out.shape[1] == 3: colors = ["r", "b", "g", "k"]
                elif out.shape[1] == 2: colors = ["b", "g", "k"]
                elif out.shape[1] == 1: colors = ["b", "k"]
                N = out.shape[1] + 1
                values = out[0].tolist() + [c_out[i].item()]

                ax2.bar(np.arange(i*w + N*i, i*w + N*(i+1)), values, color=colors)
        else:
            ax2.bar(np.arange(0, c_out.shape[0]), c_out, color="k")


        plt.show()


args = parser.parse_args()

model = ClassifierModule.load_from_checkpoint(args.model)

model.eval()
model.freeze()
if torch.cuda.is_available(): model.cuda()

dataset = None
if args.dataset == "mnist":
    _, _, dataset, _ = get_mnist_split(1)
if args.dataset == "fmnist":
    _, _, dataset, _ = get_fmnist_split(1)
if args.dataset == "kmnist":
    _, _, dataset, _ = get_kmnist_split(1)

plot(model, dataset, "cuda:0" if torch.cuda.is_available() else "cpu", args)
