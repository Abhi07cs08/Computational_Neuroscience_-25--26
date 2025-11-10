import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50


class ProjectionHead(nn.Module):
    def __init__(self, in_dim, hid=2048, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid),
            nn.ReLU(inplace=True),
            nn.Linear(hid, out_dim),
        )

    def forward(self, x):
        # ensure contiguous before Linear
        return self.net(x.contiguous())


class SimCLR(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        base = resnet50(weights=None)
        feat_dim = base.fc.in_features
        base.fc = nn.Identity()
        self.encoder = base
        self.proj = ProjectionHead(feat_dim, 2048, out_dim)

    def forward(self, x):
        # avoid channels_last; keep contiguous to dodge .view() stride issues
        x = x.contiguous()
        h = self.encoder(x)
        h = h.contiguous()
        z = self.proj(h)
        z = F.normalize(z, dim=1)
        return z
