import torch
import numpy as np

def spectral_loss(activation, device=None, n_components=40, target_alpha=1.0, bounds=(6, 30), eps=1e-12):
    X = activation.view(activation.size(0), -1)
    # print(f"Activation shape after view: {X.shape}")
    X = X - X.mean(axis=0)
    B, N = X.shape
    q=min(n_components, B, N)
    with torch.amp.autocast(enabled=False, device_type=device):
        X32 = X.float()
        U, S, V = torch.pca_lowrank(X32, q=q)
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
    return loss_value if device is None else loss_value.to(device), alpha

def just_alpha(activation, device=None, n_components=40, bounds=(6, 30), eps=1e-12):
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
    return alpha if device is None else alpha.to(device)

if __name__ == "__main__":
    from src.models.simclr_model import SimCLR
    from src.utils.model_activations import ModelActivations
    model = SimCLR(out_dim=128).to('cpu')
    activationclass = ModelActivations(model, layers=['encoder.layer4.0.bn1'])
    activationclass.register_hooks()
    x = torch.randn(4, 3, 128, 128).to('cpu')
    z = model(x)
    print("Output shape:", z.shape)
    print(z.max(), z.min())
    print(z.mean())
    acts = activationclass.fetch_activations('encoder.layer4.0.bn1')
    print("Fetched activation shape:", acts.shape)
    loss_value, alpha = spectral_loss(acts, device='cpu')
    print("Spectral loss value:", loss_value.item())
    print("Alpha:", alpha.item())
    # acts = torch.randn(16, 128, 8, 8)
    # loss_value, alpha = spectral_loss(acts, device='cpu')
    # print("Spectral loss value:", loss_value.item())
    # print("Alpha:", alpha.item())