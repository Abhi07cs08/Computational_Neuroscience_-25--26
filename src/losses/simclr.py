import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F

def info_nce(z1, z2, tau=0.2):
    """
    Stable InfoNCE under AMP:
    - compute similarities in float32
    - use dtype-safe mask fill
    """
    # normalize in float32 for stability, this should resolve the runtime error.
    z1 = F.normalize(z1.float(), dim=1)
    z2 = F.normalize(z2.float(), dim=1)

    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)                 # [2B, D]
    sim = (z @ z.t()) / tau                        # float32

    # mask self-similarity
    mask = torch.eye(2 * B, device=sim.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, torch.finfo(sim.dtype).min)

    # positives: i <-> i+B
    pos = torch.cat([torch.arange(B, 2*B, device=sim.device),
                     torch.arange(0, B, device=sim.device)], dim=0)
    labels = pos

    loss = F.cross_entropy(sim, labels)
    return loss
