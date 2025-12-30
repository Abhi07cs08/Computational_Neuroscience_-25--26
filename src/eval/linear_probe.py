import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.no_grad()
def _extract_features(encoder, dl, device):
    encoder.eval()
    feats, ys = [], []
    for x, y in dl:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        h = encoder(x)              # IMPORTANT: use encoder features, not projection z
        feats.append(h.detach())
        ys.append(y)
    return torch.cat(feats, 0), torch.cat(ys, 0)


def linear_probe_top1(encoder, tr_dl, va_dl, num_classes, device,
                      epochs=10, lr=0.1, wd=0.0, feat_batch=2048):
    """
    Freeze encoder, train a single Linear layer on frozen encoder features.
    Returns val top-1 (%).
    """
    with torch.no_grad():
        Xtr, ytr = _extract_features(encoder, tr_dl, device)
        Xva, yva = _extract_features(encoder, va_dl, device)

    feat_dim = Xtr.shape[1]
    clf = nn.Linear(feat_dim, num_classes).to(device)
    opt = torch.optim.SGD(clf.parameters(), lr=lr, momentum=0.9, weight_decay=wd)

    n = Xtr.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        Xtr_shuf = Xtr[perm]
        ytr_shuf = ytr[perm]

        clf.train()
        for i in range(0, n, feat_batch):
            xb = Xtr_shuf[i:i+feat_batch]
            yb = ytr_shuf[i:i+feat_batch]
            loss = F.cross_entropy(clf(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    clf.eval()
    with torch.no_grad():
        pred = clf(Xva).argmax(dim=1)
        return (pred == yva).float().mean().item() * 100.0
