import torch
import torch.nn.functional as F

@torch.no_grad()
def knn_top1(encoder, tr_dl, va_dl, device, k=1, temperature=0.07):
    """
    kNN on encoder features.
    For k=1: nearest neighbor.
    For k>1: similarity-weighted vote (standard in SSL eval).
    """
    encoder.eval()

    feats, labels = [], []
    for x, y in tr_dl:
        x = x.to(device, non_blocking=True)
        h = F.normalize(encoder(x), dim=1)
        feats.append(h)
        labels.append(y.to(device, non_blocking=True))
    feats = torch.cat(feats, 0)     # [N, D]
    labels = torch.cat(labels, 0)   # [N]
    ft = feats.t()                  # [D, N]

    num_classes = int(labels.max().item()) + 1

    correct = total = 0
    for x, y in va_dl:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        h = F.normalize(encoder(x), dim=1)  # [B, D]
        sims = (h @ ft)                     # [B, N]

        if k == 1:
            nn = sims.argmax(dim=1)
            pred = labels[nn]
        else:
            vals, idx = sims.topk(k=k, dim=1, largest=True, sorted=True)  # [B,k]
            nbr_labels = labels[idx]                                      # [B,k]
            weights = (vals / temperature).exp()                          # [B,k]

            scores = torch.zeros(h.size(0), num_classes, device=device)
            scores.scatter_add_(1, nbr_labels, weights)
            pred = scores.argmax(dim=1)

        correct += (pred == y).sum().item()
        total += y.numel()

    return 100.0 * correct / max(1, total)
