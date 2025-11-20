import torch
import numpy as np

def spectral_loss(activation, device=None, n_components=40, target_alpha=1.0, bounds=(10, 30), eps=1e-12):
    X = activation.view(activation.size(0), -1)
    X = X - X.mean(axis=0)
    B, N = X.shape
    q=min(n_components, B, N)
    U, S, V = torch.pca_lowrank(X, q=q)
    eigvals = S**2
    eigvals = eigvals / (torch.sum(eigvals) + eps)
    min_idx = max(1, int(bounds[0]))
    max_idx = min(int(bounds[1]), eigvals.shape[-1])
    idx = torch.arange(min_idx, max_idx, device=eigvals.device, dtype=eigvals.dtype)
    y = torch.log(eigvals[min_idx:max_idx] + eps)
    x = torch.log(idx + 0.0)

    x = (x - x.mean()) / (x.std() + eps)
    y = y - y.mean(dim=-1, keepdim=True)
    slope = (y * x).mean(dim=-1) / (x.pow(2).mean() + eps)
    alpha = -slope
    loss_value = (alpha - target_alpha).abs().mean()
    return loss_value if device is None else loss_value.to(device)