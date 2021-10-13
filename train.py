from argparse import ArgumentParser
import torch
import pytorch_lightning as pl
import os
import numpy as np
import time

from classifier import ClassifierModule

parser = ArgumentParser(add_help=False)

parser.add_argument("--model", type=str, default="nbrf", choices=["nbrf", "cnn"])

parser.add_argument("--dry", default=False, action="store_true")
parser.add_argument("--logdir", type=str, default="./logs")
parser.add_argument("--gpus", type=int, default=1)
parser.add_argument("--logfile", type=str, default="train_results.csv")

parser.add_argument("--dataset", type=str, required=True, choices=["mnist","fmnist", "kmnist"])

parser.add_argument("--noise", type=str, default="batch", choices=["dataset","batch", "white"])
parser.add_argument("--noise-position", type=str, default="backbone", choices=["head","backbone"])

parser.add_argument("--without-noise", default=False, action="store_true")
parser.add_argument("--without-others", default=False, action="store_true")

parser.add_argument("--lr", type=float, default=.001)
parser.add_argument("--batch-size", type=int, default=100)

parser.add_argument("--experiment", type=str, default="")
parser.add_argument("--wandb", action="store_true")
parser.add_argument("--wandb-entity", type=str, default="mlpi")
parser.add_argument("--wandb-project", type=str, default="nbrf")

args = parser.parse_args()

vars(args)["with_noise"] = not args.without_noise
vars(args)["with_others"] = not args.without_others

model = ClassifierModule(args)

print(f"Cuda available: {torch.cuda.is_available()}")
print(f"Device count: {torch.cuda.device_count()}")
print(f"Current device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
print(f"Current device number: {torch.cuda.current_device()}\n")

callbacks = []
early_stop_callback = pl.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0.00,
    patience=3,
    verbose=False,
    mode='min'
)
callbacks.append(early_stop_callback)

loggers = []
if args.wandb:
    from pytorch_lightning.loggers import WandbLogger
    wandb_logger = WandbLogger(entity=args.wandb_entity, project=args.wandb_project)
    loggers.append(wandb_logger)

trainer = pl.Trainer(
    weights_summary='full',
    gpus=args.gpus,
    callbacks=callbacks,
    fast_dev_run=args.dry,
    weights_save_path=args.logdir,
    logger=loggers,
    log_every_n_steps=1
)

trainer.fit(model)
trainer.test(model)

