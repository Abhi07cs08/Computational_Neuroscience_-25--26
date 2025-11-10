import torch
import torch.nn.functional as F

def info_nce(z1, z2, tau=0.1):
    # z1,z2 are already L2-normalized by the model
    B, D = z1.shape
    Z = torch.cat([z1, z2], dim=0)             # [2B, D]
    sim = Z @ Z.t() / tau                      # cosine similarities / tau

    # mask self-similarities
    mask = torch.eye(2*B, device=Z.device).bool()
    sim = sim.masked_fill(mask, -1e9)

    # positives: i <-> i+B and i+B <-> i
    pos = torch.cat([torch.arange(B, 2*B), torch.arange(0, B)], dim=0).to(Z.device)
    labels = pos
    loss = F.cross_entropy(sim, labels)
    return loss
