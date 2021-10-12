from argparse import ArgumentParser
import pytorch_lightning as pl
import os
import numpy as np
import time

from systems.nbrf import NBRFSystem
from systems.cnn import CNNSystem

from datasets.loaders import *

from modules.backbones import MNISTCNN
from modules.heads import MNISTRF, MNISTFF
from modules.noise_generator import CyclicNoise


parser = ArgumentParser(add_help=False)

parser.add_argument("--model", type=str, default="nbrf", choices=["nbrf", "cnn"])

parser.add_argument("--dry", default=False, action="store_true")
parser.add_argument("--logdir", type=str, default=".")
parser.add_argument("--log", default=False, action="store_true")
parser.add_argument("--logfile", type=str, default="train_results.csv")

parser.add_argument("--dataset", type=str, required=True, choices=["mnist","fmnist", "kmnist"])

parser.add_argument("--noise", type=str, default="batch", choices=["dataset","batch", "white"])
parser.add_argument("--noise-position", type=str, default="backbone", choices=["head","backbone"])

parser.add_argument("--without-noise", default=False, action="store_true")
parser.add_argument("--without-other", default=False, action="store_true")

parser.add_argument("--name", type=str, default="test")
parser.add_argument("--lr", type=float, default=.001)
parser.add_argument("--batch-size", type=int, default=100)

parser.add_argument("--n-outputs", type=int, default=10)

parser.add_argument("--experiment", type=str, default="")
parser.add_argument("--wandb", action="store_true")
parser.add_argument("--wandb-entity", type=str, default="mlpi")
parser.add_argument("--wandb-project", type=str, default="nbrf")


args = parser.parse_args()
dargs = vars(args)

dargs["with_noise"] = not args.without_noise
dargs["with_other"] = not args.without_other

if args.model == "nbrf":
    dargs["backbone"] = MNISTCNN
    dargs["head"] = MNISTRF
    dargs["n_rfs"] = 10

if args.model == "cnn":
    dargs["backbone"] = MNISTCNN
    dargs["head"] = MNISTFF

if args.dataset == "mnist":
    train_loader, valid_loader, test_loader, noise_loader = get_mnist_split(args.batch_size)
if args.dataset == "fmnist":
    train_loader, valid_loader, test_loader, noise_loader = get_fmnist_split(args.batch_size)
if args.dataset == "kmnist":
    train_loader, valid_loader, test_loader, noise_loader = get_kmnist_split(args.batch_size)


dargs["train_dataset"] = train_loader
dargs["test_dataset"] = test_loader
dargs["valid_dataset"] = valid_loader
dargs["noise_generator"] = CyclicNoise(noise_loader)

if args.model == "nbrf":
    model = NBRFSystem(args)
if args.model == "cnn":
    model = CNNSystem(args)


early_stop_callback = pl.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0.00,
    patience=3,
    verbose=False,
    mode='min'
)

print(f"Cuda available: {torch.cuda.is_available()}")
print(f"Device count: {torch.cuda.device_count()}")
print(f"Current device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
print(f"Current device number: {torch.cuda.current_device()}\n")

loggers = []
if args.wandb:
    import wandb
    wandb.init(config=args, project=args.wandb_project, entity=args.wandb_entity)
    wandb.watch(model, log='all', log_freq=1)
    wandb_logger = pl.loggers.WandbLogger()
    wandb_logger.log_hyperparams(args)
    loggers.append(wandb_logger)

trainer = pl.Trainer(
    weights_summary='full',
    gpus=1,
    callbacks=[early_stop_callback],
    fast_dev_run=args.dry,
    weights_save_path=args.logdir,
    logger=loggers,
    log_every_n_steps=1
)

print("Training start")
t0 = time.time()
trainer.fit(model)
print(f"Time elapsed: {time.time() - t0}\n")
trainer.test(model)
if args.log:
    model.manual_log(args.logfile)

if args.wandb:
    wandb.finish()
