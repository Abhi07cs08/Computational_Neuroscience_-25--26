# src/utils/spectrum.py
import torch

@torch.no_grad()
def spectrum_stats(model, loader, device, batches=10):
    """
    Collect ~batches of embeddings, compute covariance spectrum,
    and return (ParticipationRatio, smallest_eigenvalue, top64_energy_fraction).
    """
    model.eval()
    zs = []
    it = iter(loader)
    for b in range(batches):
        try:
            q, _ = next(it)  # one view is enough
        except StopIteration:
            break
        q = q.to(device, non_blocking=True)
        z = model(q)               # [B, D], L2-normalized already
        zs.append(z.detach().cpu())

    if len(zs) == 0:
        model.train()
        return 0.0, 0.0, 0.0

    Z = torch.cat(zs, dim=0)       # [N, D]
    Z = Z - Z.mean(dim=0, keepdim=True)
    C = (Z.T @ Z) / Z.shape[0]     # [D, D]
    evals = torch.linalg.eigvalsh(C).clamp_min(1e-12)  # ascending
    lam = torch.flip(evals, dims=(0,))                 # descending

    # Participation Ratio (effective dimensionality proxy)
    pr = (lam.sum() ** 2 / lam.pow(2).sum()).item()
    lam_min = lam[-1].item()
    topk = lam[:64].sum().item() / lam.sum().item()

    model.train()
    return pr, lam_min, topk
