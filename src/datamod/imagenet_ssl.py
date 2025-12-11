import os
import warnings
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from src.datamod.twocrops import TwoCropsTransform, simclr_transform
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # prevents hang on bad JPEGs
from torchvision import transforms


# Safe collate to drop None samples
def _safe_collate(batch):
    batch = [b for b in batch if b is not None and b[0] is not None and b[1] is not None]
    if len(batch) == 0:
        raise RuntimeError("All samples were None in this batch.")
    q = torch.stack([b[0] for b in batch], dim=0)
    k = torch.stack([b[1] for b in batch], dim=0)
    return q, k


# Dataset class that skips corrupted images instead of crashing
class SafeImageFolder(ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        try:
            sample = self.loader(path)
            if self.transform is not None:
                sample = self.transform(sample)
            return sample
        except Exception as e:
            warnings.warn(f"[WARN] Skipping {path}: {e}")
            return None


# ---- Main dataloader builder ----
def build_imagenet_loaders(
    root,
    batch_size=256,
    workers=0,
    img_size=224,
    pin_memory=False,
    limit_train=None,
):
    """
    Args:
        root: path to dataset root (expects /train and /val folders)
        batch_size: int
        workers: int
        img_size: resize crop
        pin_memory: bool (True only for CUDA)
        limit_train: int or None, number of samples to load for bring-up
    """
    t = TwoCropsTransform(simclr_transform(img_size))
    train_path = os.path.join(root, "train")
    dataset = SafeImageFolder(root=train_path, transform=t)

    # Optional: load only a subset for debugging or small experiments
    if limit_train is not None and limit_train < len(dataset):
        indices = list(range(limit_train))
        dataset = Subset(dataset, indices)

    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin_memory,
        drop_last=True,
        collate_fn=_safe_collate,
    )
    return dl

def build_imagenet_val_loader(
    root,
    batch_size=256,
    workers=0,
    img_size=224,
    pin_memory=False,
    limit_val=None,
):
    """
    Args:
        root: path to dataset root (expects /val folder)
        batch_size: int
        workers: int
        img_size: resize crop
        pin_memory: bool (True only for CUDA)
        limit_val: int or None, number of samples to load for bring-up
    """
    t = transforms.Compose([transforms.ToTensor(),])
    val_path = os.path.join(root, "val")
    dataset = SafeImageFolder(root=val_path, transform=t)

    if limit_val is not None and limit_val < len(dataset):
        indices = list(range(limit_val))
        dataset = Subset(dataset, indices)


    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return dl


if __name__ == "__main__":
    dl = build_imagenet_val_loader("train_val")
    x = next(iter(dl))
    print(len(x))
    print(type(x))
    print(x[0].shape, x[1].shape)