# src/eval/knn.py
import torch
import torch.nn.functional as F


@torch.no_grad()
def knn_top1(encoder, tr_dl, va_dl, device, k=1):
    """
    kNN on encoder features h, not projection z.
    For k=1, this is your quick sanity metric.
    """
    encoder.eval()

    # Memory bank
    feats, labels = [], []
    for x, y in tr_dl:
        x = x.to(device, non_blocking=True)
        h = encoder(x)
        h = F.normalize(h, dim=1)
        feats.append(h)
        labels.append(y.to(device, non_blocking=True))
    feats = torch.cat(feats, 0)        # [N, D]
    labels = torch.cat(labels, 0)      # [N]
    ft = feats.t()                     # [D, N]

    # Query
    correct = total = 0
    for x, y in va_dl:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        h = F.normalize(encoder(x), dim=1)     # [B, D]
        sims = h @ ft                          # [B, N]
        nn = sims.argmax(dim=1)                # [B]
        pred = labels[nn]
        correct += (pred == y).sum().item()
        total += y.numel()

    return 100.0 * correct / max(1, total)
