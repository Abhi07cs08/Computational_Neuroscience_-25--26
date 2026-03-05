# src/datamod/imagenet_ssl.py
import os
import torchvision
import warnings
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from PIL import ImageFile
from src.datamod.twocrops import TwoCropsTransform, simclr_transform, ssl_deterministic_transform, eval_train_transform, eval_val_transform

from src.datamod.twocrops import TwoCropsTransform, simclr_transform, eval_train_transform, eval_val_transform, simclr_transform_cifar, eval_train_transform_cifar, eval_val_transform_cifar

ImageFile.LOAD_TRUNCATED_IMAGES = True


def _safe_collate_ssl(batch):
    # batch entries are (q, k) for SSL
    batch = [b for b in batch if b is not None and b[0] is not None and b[1] is not None]
    if len(batch) == 0:
        raise RuntimeError("All samples were None in this batch.")
    q = torch.stack([b[0] for b in batch], dim=0)
    k = torch.stack([b[1] for b in batch], dim=0)
    return q, k


def _safe_collate_supervised(batch):
    # batch entries are (x, y) for supervised eval
    batch = [b for b in batch if b is not None and b[0] is not None]
    if len(batch) == 0:
        raise RuntimeError("All samples were None in this batch.")
    x = torch.stack([b[0] for b in batch], dim=0)
    y = torch.as_tensor([b[1] for b in batch], dtype=torch.long)
    return x, y


class SafeImageFolder(ImageFolder):
    """
    Returns whatever the underlying ImageFolder would return, but skips corrupted images.
    """
    def __getitem__(self, index):
        path, target = self.samples[index]
        try:
            sample = self.loader(path)
            if self.transform is not None:
                sample = self.transform(sample)

            # If transform returns (q,k) -> SSL
            if isinstance(sample, tuple) and len(sample) == 2:
                return sample

            # Else supervised-style (x, y)
            return sample, target

        except Exception as e:
            warnings.warn(f"[WARN] Skipping {path}: {e}")
            return None

def build_ssl_train_loader_cifar10(
    root,
    batch_size=256,
    workers=8,
    img_size=32,
    pin_memory=False,
    limit_train=None,):

    train_path = os.path.join(root, "train")
    t = TwoCropsTransform(simclr_transform_cifar(img_size))
    ds = SafeImageFolder(root=train_path, transform=t)
    # ds = torchvision.datasets.CIFAR10(root=root, train=True, transform=t, download=True)

    if limit_train is not None and limit_train < len(ds):
        ds = Subset(ds, list(range(limit_train)))

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory,
        drop_last=True,
        collate_fn=_safe_collate_ssl,
        persistent_workers=(workers > 0),
    )

def build_ssl_val_loader_deterministic(
    root,
    batch_size=256,
    workers=8,
    img_size=224,
    pin_memory=False,
    limit_val=None,
):
    """
    Deterministic SSL validation: two-crops but deterministic (same transform twice).
    This is a diagnostic for collapse, not the main SSL-val objective.
    """
    val_path = os.path.join(root, "val")
    base = ssl_deterministic_transform(img_size)
    t = TwoCropsTransform(base)
    ds = SafeImageFolder(root=val_path, transform=t)

    if limit_val is not None and limit_val < len(ds):
        ds = Subset(ds, list(range(limit_val)))

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory,
        drop_last=True,
        collate_fn=_safe_collate_ssl,
        persistent_workers=(workers > 0),
    )

def build_ssl_val_loader_cifar10(
    root,
    batch_size=256,
    workers=8,
    img_size=32,
    pin_memory=False,
    limit_val=None,):

    val_path = os.path.join(root, "test")
    t = TwoCropsTransform(simclr_transform_cifar(img_size))
    ds = SafeImageFolder(root=val_path, transform=t)
    if limit_val is not None and limit_val < len(ds):
        ds = Subset(ds, list(range(limit_val)))

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory,
        drop_last=True,
        collate_fn=_safe_collate_ssl,
        persistent_workers=(workers > 0),
    )

def build_eval_loaders_cifar10(
    root,
    batch_size=256,
    workers=8,
    img_size=32,
    pin_memory=False,
    limit_train=None,
    limit_val=None,):

    tr_ds = SafeImageFolder(root=os.path.join(root, "train"), transform=eval_train_transform_cifar(img_size))
    va_ds = SafeImageFolder(root=os.path.join(root, "test"), transform=eval_val_transform_cifar(img_size))
    if limit_train is not None and limit_train < len(tr_ds):
        tr_ds = Subset(tr_ds, list(range(limit_train)))
    if limit_val is not None and limit_val < len(va_ds):
        va_ds = Subset(va_ds, list(range(limit_val)))

    tr_dl = DataLoader(
        tr_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory,
        drop_last=True,
        collate_fn=_safe_collate_supervised,
        persistent_workers=(workers > 0),
    )
    va_dl = DataLoader(
        va_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory,
        drop_last=True,
        collate_fn=_safe_collate_supervised,
        persistent_workers=(workers > 0),
    )
    return tr_dl, va_dl