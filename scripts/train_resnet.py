import argparse
import os
import torch
from torch import optim
from torch.amp import GradScaler, autocast
import csv, time, os

from src.datamod.imagenet_ssl import build_imagenet_loaders
from torchvision.models import resnet18
from torchvision import transforms, datasets
import torch.nn.functional as F
import numpy as np


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--imagenet_root', type=str, required=True)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--accum_steps', type=int, default=1)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--workers', type=int, default=0)      # start with 0 on macOS
    ap.add_argument('--img_size', type=int, default=224)
    ap.add_argument('--tau', type=float, default=0.1)
    ap.add_argument('--lr', type=float, default=0.1)
    ap.add_argument('--wd', type=float, default=1e-6)
    ap.add_argument('--amp', action='store_true', help='Enable CUDA AMP only')
    ap.add_argument('--limit_train', type=int, default=2048, help='subset size for bring-up')
    ap.add_argument('--log_every', type=int, default=50)
    ap.add_argument('--beta', type=float, default=1.0, help='weight for spectral loss')
    return ap.parse_args()


def main():
    args = parse_args()

    # device preference: MPS > CUDA > CPU
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print("Using device:", device)
    # pin_memory is only useful on CUDA
    pin_memory = (device == 'cuda')

    beta = args.beta

    train_dl = build_imagenet_loaders(
        root=args.imagenet_root,
        batch_size=args.batch_size,
        workers=args.workers,
        img_size=args.img_size,
        pin_memory=pin_memory,
        limit_train=args.limit_train,
    )

    model = resnet18(pretrained=True).to(device)

    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.wd
    )

    model.train()
    for epoch in range(1, args.epochs + 1):