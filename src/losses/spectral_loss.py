from src.latest_neural_data.model_acts import extract_model_activations_from_cache
import torch
import numpy as np
from src.models.simclr_model import SimCLR
from src.utils.post_training import extract_val_dl_from_ckpt, extract_model_weights, extract_ckpt_args
from src.utils.model_activations import ModelActivations


def spectral_loss(activation, device=None, n_components=40, target_alpha=1.0, bounds=(0, 10), log_base="10", eps=1e-12):
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
    idx = torch.arange(min_idx, max_idx+1, device=eigvals.device, dtype=eigvals.dtype)
    if log_base == "e":
        log_fn = torch.log
    elif log_base == "10":
        log_fn = torch.log10
    y = log_fn(eigvals[min_idx-1:max_idx] + eps)
    x = log_fn(idx + 0.0)

    # x = (x - x.mean()) / (x.std() + eps)
    x= x - x.mean()
    y = y - y.mean(dim=-1, keepdim=True)
    slope = (y * x).mean(dim=-1) / (x.pow(2).mean() + eps)
    alpha = -slope
    loss_value = (alpha - target_alpha).abs().mean()
    return loss_value if device is None else loss_value.to(device), alpha

def just_alpha(activation, device=None, n_components=40, bounds=(0, 10), eps=1e-12, log_base="10", return_eigvals=False):
    # print(f"Activation shape: {activation.shape}")
    X = activation.view(activation.size(0), -1)
    # print(f"Activation shape after view: {X.shape}")
    X = X - X.mean(axis=0)
    B, N = X.shape
    q=min(n_components, B, N)
    U, S, V = torch.pca_lowrank(X, q=q)
    eigvals = S**2
    eigvals = eigvals / (torch.sum(eigvals) + eps)
    min_idx = max(1, int(bounds[0]))
    max_idx = min(int(bounds[1]), eigvals.shape[-1])
    idx = torch.arange(min_idx, max_idx+1, device=eigvals.device, dtype=eigvals.dtype)
    if log_base == "e":
        log_fn = torch.log
    elif log_base == "10":
        log_fn = torch.log10
    y = log_fn(eigvals[min_idx-1:max_idx] + eps)
    x = log_fn(idx + 0.0)

    x = (x - x.mean()) / (x.std() + eps)
    y = y - y.mean(dim=-1, keepdim=True)
    slope = (y * x).mean(dim=-1) / (x.pow(2).mean() + eps)
    alpha = -slope
    if return_eigvals:
        return alpha if device is None else alpha.to(device), eigvals
    return alpha if device is None else alpha.to(device)

def just_alpha_fixed(activation, device=None, n_components=40, bounds=(0, 10), eps=1e-12, log_base="10",return_eigvals=False):
    X = activation.view(activation.size(0), -1)
    # print(f"Activation shape after view: {X.shape}")
    X = X - X.mean(axis=0)
    B, N = X.shape
    q=min(n_components, B, N)
    U, S, V = torch.pca_lowrank(X, q=q)
    eigvals = S**2
    eigvals = eigvals / (torch.sum(eigvals) + eps)

    min_idx = max(1, int(bounds[0]))
    max_idx = min(int(bounds[1]), eigvals.shape[-1])

    # 1-based PC indices, inclusive range
    idx = torch.arange(min_idx, max_idx + 1, device=eigvals.device, dtype=eigvals.dtype)
    if log_base == "e":
        log_fn = torch.log
    elif log_base == "10":
        log_fn = torch.log10
    y = log_fn(eigvals[min_idx - 1:max_idx] + eps)
    x = log_fn(idx + 0.0)

    # center only, do not standardize x
    x = x - x.mean()
    y = y - y.mean(dim=-1, keepdim=True)

    slope = (y * x).mean(dim=-1) / (x.pow(2).mean() + eps)
    alpha = -slope
    if return_eigvals:
        return alpha if device is None else alpha.to(device), eigvals
    return alpha if device is None else alpha.to(device)

def just_alpha_imgnet_standalone(ckpt_path, dl_kwargs = {"workers": 3}, alpha_kwargs={}):
    args = extract_ckpt_args(ckpt_path, as_args=True)
    eval_tr_dl, eval_va_dl = extract_val_dl_from_ckpt(ckpt_path, kwargs=dl_kwargs)
    base_ds = eval_tr_dl.dataset.dataset if hasattr(eval_tr_dl.dataset, "dataset") else eval_tr_dl.dataset
    num_classes = len(base_ds.classes)
    model = SimCLR()
    state_dict = extract_model_weights(ckpt_path)
    model.load_state_dict(state_dict)
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:        device = "cpu"
    model = model.to(device)
    for key, value in alpha_kwargs.items():
        setattr(args, key, value)

    activationclass = ModelActivations(model, layers=[args.neural_ev_layer])
    activationclass.register_hooks()
    alphas = []
    with torch.no_grad():
        for batch in eval_tr_dl:
            inputs = batch[0].to(device)
            _ = model(inputs)
            a = just_alpha_fixed(activationclass.activations[args.neural_ev_layer], device=device)
            alphas.append(float(a.detach().float().cpu().item()))
    avg_alpha = float(np.mean(alphas))
    return avg_alpha

def just_alpha_brainscore_standalone(ckpt_path, dl_kwargs = {"workers": 3}, alpha_kwargs={}):
    args = extract_ckpt_args(ckpt_path, as_args=True)
    eval_tr_dl, eval_va_dl = extract_val_dl_from_ckpt(ckpt_path, kwargs=dl_kwargs)
    base_ds = eval_tr_dl.dataset.dataset if hasattr(eval_tr_dl.dataset, "dataset") else eval_tr_dl.dataset
    num_classes = len(base_ds.classes)
    model = SimCLR()
    state_dict = extract_model_weights(ckpt_path)
    model.load_state_dict(state_dict)
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:        device = "cpu"
    model = model.to(device)
    for key, value in alpha_kwargs.items():
        setattr(args, key, value)

    model_activations, stimulus_ids = extract_model_activations_from_cache(
            model=model,
            cache_dir=args.neural_data_dir,
            layer_name=args.neural_ev_layer,
            batch_size=args.batch_size
        )
    model_activations = torch.tensor(model_activations) if not isinstance(model_activations, torch.Tensor) else model_activations
    alphas = []
    a = just_alpha_fixed(model_activations.to(device), device=device)
    return a.item()
    

        

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