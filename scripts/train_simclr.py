import argparse
import os
import torch
from torch import optim
from torch.amp import GradScaler, autocast
import csv, time, os

from src.datamod.imagenet_ssl import build_imagenet_loaders
from src.models.simclr_model import SimCLR
from src.losses.simclr import info_nce
from torchvision import transforms, datasets
import torch.nn.functional as F
import numpy as np

@torch.no_grad()
def knn_top1_fast(model, root, img_size=128, train_samples=2000, val_samples=500, batch=64):
    """
    Fast 1-NN sanity check:
      - Subsamples train/val to keep it quick
      - Uses batched dataloaders
      - Runs entirely on the model's device
    """
    from torchvision import transforms, datasets
    import torch

    device = next(model.parameters()).device

    t = transforms.Compose([
        transforms.Resize(int(img_size*1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    tr = datasets.ImageFolder(os.path.join(root, "train"), transform=t)
    va = datasets.ImageFolder(os.path.join(root, "val"),   transform=t)

    g = torch.Generator().manual_seed(0)
    tr_idx = torch.randperm(len(tr), generator=g)[:train_samples]
    va_idx = torch.randperm(len(va), generator=g)[:val_samples]

    tr_dl = torch.utils.data.DataLoader(
        tr, batch_size=batch, sampler=torch.utils.data.SubsetRandomSampler(tr_idx),
        num_workers=0, pin_memory=False
    )
    va_dl = torch.utils.data.DataLoader(
        va, batch_size=batch, sampler=torch.utils.data.SubsetRandomSampler(va_idx),
        num_workers=0, pin_memory=False
    )

    model.eval()

    # Build memory bank (train feats)
    feats_list, labels_list = [], []
    for imgs, ys in tr_dl:
        imgs = imgs.to(device, non_blocking=False)
        h = model.encoder(imgs)
        z = model.proj(h)
        z = F.normalize(z, dim=1)
        feats_list.append(z)           # keep on device
        labels_list.append(ys.to(device))

    feats = torch.cat(feats_list, 0)    # [Nt, D] on device
    labels = torch.cat(labels_list, 0)  # [Nt] on device
    ft = feats.t()                      # [D, Nt] on device

    # Query (val feats)
    correct = total = 0
    for imgs, ys in va_dl:
        imgs = imgs.to(device, non_blocking=False)
        ys   = ys.to(device)
        h = model.encoder(imgs)
        z = model.proj(h)
        z = F.normalize(z, dim=1)       # [B, D] on device
        sims = z @ ft                   # [B, Nt]
        nn_idx = sims.argmax(dim=1)
        pred = labels[nn_idx]
        correct += (pred == ys).sum().item()
        total   += ys.numel()

    acc = 100.0 * correct / max(1, total)
    model.train()
    return acc


@torch.no_grad()
def spectrum_probe(model, dl, batches=4):
    zs = []
    it = 0
    for q,k in dl:
        q = q[:32].to(next(model.parameters()).device)  # small sample
        z = model(q)
        zs.append(z.cpu())
        it += 1
        if it >= batches: break
    Z = torch.cat(zs, 0).numpy()       # [N, D]
    Z = Z - Z.mean(0, keepdims=True)
    C = (Z.T @ Z) / Z.shape[0]         # covariance
    eig = np.linalg.eigvalsh(C)[::-1]  # descending
    pr  = (eig.sum()**2) / ( (eig**2).sum() + 1e-12)
    return float(pr), eig[0], eig[-1]


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

    # pin_memory is only useful on CUDA
    pin_memory = (device == 'cuda')

    train_dl = build_imagenet_loaders(
        root=args.imagenet_root,
        batch_size=args.batch_size,
        workers=args.workers,
        img_size=args.img_size,
        pin_memory=pin_memory,
        limit_train=args.limit_train,
    )

    model = SimCLR(out_dim=128).to(device)

    optimizer = optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd
    )

    # AMP only on CUDA
    amp_enabled = (args.amp and device == 'cuda')
    scaler = GradScaler('cuda') if amp_enabled else GradScaler('cpu')

    model.train()
    for epoch in range(1, args.epochs + 1):
        running = 0.0
        optimizer.zero_grad(set_to_none=True)

        for it, (q, k) in enumerate(train_dl):
            # keep tensors contiguous
            q = q.to(device, non_blocking=False).contiguous()
            k = k.to(device, non_blocking=False).contiguous()

            if amp_enabled:
                with autocast('cuda'):
                    z1 = model(q)
                    z2 = model(k)
                    loss = info_nce(z1, z2, tau=args.tau) / args.accum_steps
                scaler.scale(loss).backward()
            else:
                z1 = model(q)
                z2 = model(k)
                loss = info_nce(z1, z2, tau=args.tau) / args.accum_steps
                loss.backward()

            if (it + 1) % args.accum_steps == 0:
                if amp_enabled:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            running += loss.item() * args.accum_steps  # undo the division for logging

            if (it + 1) % args.log_every == 0:
                print(f"epoch {epoch} | iter {it+1}/{len(train_dl)} | loss {running / (it+1):.4f}")
        # --- save checkpoint ---
        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "opt": optimizer.state_dict(),
            "args": vars(args),
        }
        os.makedirs("ckpts/simclr", exist_ok=True)
        torch.save(ckpt, f"ckpts/simclr/e{epoch:03d}.pt")

        avg = running / max(1, len(train_dl))
        print(f"epoch {epoch} | avg loss {avg:.4f}")
        top1 = knn_top1_fast(model, args.imagenet_root, img_size=args.img_size,
                            train_samples=2000, val_samples=500, batch=64)
        print(f"epoch {epoch} | 1-NN top1 ~ {top1:.1f}%")
        pr, lam1, lam_min = spectrum_probe(model, train_dl, batches=2)
        print(f"epoch {epoch} | PR {pr:.1f} | lam1 {lam1:.3g} | lam_min {lam_min:.3g}")


        os.makedirs("logs", exist_ok=True)
        log_path = "logs/simclr_baseline.csv"
        header = ["ts","epoch","tau","batch_size","img_size","accum_steps","lr",
                "loss","knn_top1","PR","lam1","lam_min","limit_train","device"]
        row = [int(time.time()), epoch, args.tau, args.batch_size, args.img_size,
            args.accum_steps, args.lr, avg, top1, pr, lam1, lam_min, args.limit_train, device]
        write_header = not os.path.exists(log_path)
        with open(log_path, "a", newline="") as f:
            w = csv.writer(f)
            if write_header: w.writerow(header)
            w.writerow(row)

if __name__ == '__main__':
    main()
