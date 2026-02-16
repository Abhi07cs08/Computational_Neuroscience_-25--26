import torch
import torch.nn.functional as F

def compute_affinity(t: torch.Tensor) -> torch.Tensor:
    """
    Compute semantic affinity matrix r_ij ∈ [0,1] from teacher features.

    Args:
        t: [N, D] normalized or unnormalized teacher features

    Returns:
        r: [N, N] affinity matrix with diagonal zeroed
    """
    t = F.normalize(t.float(), dim=1)
    r = (t @ t.t()).clamp(-1.0, 1.0)
    r = 0.5 * (r + 1.0)
    diag = torch.eye(r.size(0), device=r.device, dtype=torch.bool)
    r = r.masked_fill(diag, 0.0)
    return r


@torch.no_grad()
def affinity_stats(
    r: torch.Tensor,
    threshold: float = 0.8,
) -> dict:
    """
    Compute near-positive statistics from affinity matrix.

    A near-positive is defined as r_ij > threshold, j != i.

    Returns:
        dict with:
            mean_r
            median_r
            tail_r_95
            mean_near_pos
            frac_near_pos
    """
    N = r.size(0)
    flat = r[r > 0]
    near = (r > threshold).float()
    mean_near = near.sum(dim=1).mean()
    frac_near = mean_near / max(1.0, float(N - 1))

    return {
        "mean_r": flat.mean().item() if flat.numel() > 0 else 0.0,
        "median_r": flat.median().item() if flat.numel() > 0 else 0.0,
        "tail_r_95": torch.quantile(flat.float(), 0.95).item() if flat.numel() > 0 else 0.0,
        "mean_near_pos": mean_near.item(),
        "frac_near_pos": frac_near,
    }


def info_nce(z1: torch.Tensor, z2: torch.Tensor, tau: float = 0.2) -> torch.Tensor:
    """Stable SimCLR InfoNCE under AMP.

    Implementation detail:
    - Normalize + similarity in float32 to avoid fp16 overflow.
    - Use dtype-safe mask fill.

    Args:
        z1: [B, D]
        z2: [B, D]
        tau: temperature

    Returns:
        scalar loss
    """
    z1 = F.normalize(z1.float(), dim=1)
    z2 = F.normalize(z2.float(), dim=1)

    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)  # [2B, D]

    logits = (z @ z.t()) / float(tau)  # float32

    # mask self-similarity
    diag = torch.eye(2 * B, device=logits.device, dtype=torch.bool)
    logits = logits.masked_fill(diag, torch.finfo(logits.dtype).min)

    # positives: i <-> i+B
    labels = torch.cat(
        [torch.arange(B, 2 * B, device=logits.device), torch.arange(0, B, device=logits.device)],
        dim=0,
    )

    return F.cross_entropy(logits, labels)


def debiased_info_nce(
    z1: torch.Tensor,
    z2: torch.Tensor,
    t1: torch.Tensor,
    t2: torch.Tensor,
    tau: float = 0.2,
    gamma: float = 1.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Denominator-only debiased InfoNCE:
      L_i = -log exp(sim(i,i+)/tau) / sum_{j != i} w_ij * exp(sim(i,j)/tau)

    We implement this by:
      logits = sim/tau
      logits_neg = logits + log(w_ij)  (only for j != i and j != pos(i))
      logits_pos = logits (no weight)
      diag = -inf
    """
    # student projections in float32 for AMP safety
    z1 = F.normalize(z1.float(), dim=1)
    z2 = F.normalize(z2.float(), dim=1)

    # teacher features used only for affinity
    t1 = F.normalize(t1.float(), dim=1)
    t2 = F.normalize(t2.float(), dim=1)

    B = z1.size(0)
    N = 2 * B

    z = torch.cat([z1, z2], dim=0)  # [2B, D]
    t = torch.cat([t1, t2], dim=0)  # [2B, D]

    logits = (z @ z.t()) / float(tau)  # [2B, 2B] float32

    # labels: i <-> i+B
    labels = torch.cat(
        [torch.arange(B, 2 * B, device=logits.device),
         torch.arange(0, B, device=logits.device)],
        dim=0,
    )

    # affinity in [0,1] before weighting
    r = compute_affinity(t)  # diag already 0

    # raw denominator weights (no normalization!)
    w = (1.0 - r).clamp(min=0.0).pow(float(gamma))  # [N,N]

    # build masks
    diag = torch.eye(N, device=logits.device, dtype=torch.bool)
    rows = torch.arange(N, device=logits.device)

    # never include self
    logits = logits.masked_fill(diag, torch.finfo(logits.dtype).min)
    w = w.masked_fill(diag, 0.0)

    # DO NOT weight the positive term: set w_pos = 1  (logw_pos = 0)
    w[rows, labels] = 1.0

    # add log-weights to logits (safe)
    logw = torch.log(w + eps)

    # Only denominator gets weights.
    weighted_logits = logits + logw

    return F.cross_entropy(weighted_logits, labels)
