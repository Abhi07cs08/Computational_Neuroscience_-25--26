# scripts/train_simclr.py
import argparse
import os
import time
import csv
import random
import numpy as np

import torch
import torch.nn.functional as F
from torch import optim
from torch.amp import GradScaler, autocast

from src.datamod.imagenet_ssl import (
    build_ssl_train_loader,
    build_ssl_val_loader,
    build_eval_loaders,
)
from src.models.simclr_model import SimCLR
from src.losses.simclr import info_nce
from src.losses.spectral_loss import spectral_loss, just_alpha
from src.utils.model_activations import ModelActivations

from src.eval.knn import knn_top1
from src.eval.linear_probe import linear_probe_top1

from src.metrics.FR_EV_offline import F_R_EV, three_channel_transform

from src.REVERSE_PRED_FINAL.ev_helper import forward_ev, reverse_ev
from src.REVERSE_PRED_FINAL.model_acts import extract_model_activations_from_cache



# --------------------------
# Utilities
# --------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cosine_lr_with_warmup(
    optimizer,
    base_lr,
    epoch,
    step_in_epoch,
    steps_per_epoch,
    warmup_epochs,
    total_epochs,
):
    """
    Per-step cosine schedule with linear warmup.
    """
    global_step = epoch * steps_per_epoch + step_in_epoch
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch

    if warmup_steps > 0 and global_step < warmup_steps:
        lr = base_lr * float(global_step + 1) / float(warmup_steps)
    else:
        t = (global_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        lr = 0.5 * base_lr * (1.0 + np.cos(np.pi * t))

    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


@torch.no_grad()
def ssl_val_infonce_and_alpha(model, ssl_val_dl, tau, device,
                              activationclass=None, alpha_layer=None,
                              compute_alpha=True):
    model.eval()
    total_loss = 0.0
    n = 0
    alphas = []

    for q, k in ssl_val_dl:
        q = q.to(device, non_blocking=True).contiguous()
        k = k.to(device, non_blocking=True).contiguous()

        z1 = model(q)
        z2 = model(k)
        loss = info_nce(z1, z2, tau=tau)

        total_loss += loss.item()
        n += 1

        if compute_alpha and activationclass is not None and alpha_layer is not None:
            a = just_alpha(activationclass.activations[alpha_layer], device=device)
            alphas.append(float(a.detach().float().cpu().item()))

    model.train()
    avg_loss = total_loss / max(1, n)
    avg_alpha = float(np.mean(alphas)) if (compute_alpha and len(alphas) > 0) else 0.0
    return avg_loss, avg_alpha



@torch.no_grad()
def spectrum_pr(model, dl, device, batches=2, max_per_batch=64, use="z"):
    """
    Participation Ratio (PR) + eigen stats from a few batches.
    Mirrors old spectrum_probe, but uses torch end-to-end.
    use:
      - "z": projection output (matches old model(q))
      - "h": encoder features
    """
    model.eval()
    ys = []
    it = 0

    for batch in dl:
        # SSL: (q,k) ; supervised eval: (x,y)
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            x = batch[0]
        else:
            x = batch

        x = x[:max_per_batch].to(device, non_blocking=True).contiguous()

        h = model.encoder(x)
        if use == "z":
            y = model.proj(h)
            y = F.normalize(y, dim=1)
        else:
            y = F.normalize(h, dim=1)

        ys.append(y.detach().float().cpu())
        it += 1
        if it >= batches:
            break

    if len(ys) == 0:
        model.train()
        return 0.0, 0.0, 0.0

    Y = torch.cat(ys, dim=0)  # [N, D] on CPU
    Y = Y - Y.mean(dim=0, keepdim=True)
    C = (Y.T @ Y) / max(1, Y.shape[0])  # [D, D]
    eig = torch.linalg.eigvalsh(C)      # ascending
    eig = torch.clamp(eig, min=0)

    pr = (eig.sum() ** 2) / (eig.pow(2).sum() + 1e-12)
    lam1 = eig[-1]
    lam_min = eig[0]

    model.train()
    return float(pr.item()), float(lam1.item()), float(lam_min.item())

# def parse_args_ckpt():
#     ap = argparse.ArgumentParser()    
    
#     ap.add_argument("--ckpt_path", type=str, required=True)

#     return ap.parse_args()


def main(imagenet_root=None, epochs=300, batch_size=512, img_size=224, tau=0.2, lr=0.3, wd=1e-6, workers=8,
         accum_steps=1, warmup_epochs=10, amp=False, grad_clip=0.0, limit_train=None, limit_val=None,
         log_every=50, save_dir="", skip_knn=False, skip_alpha=False, skip_neural_ev=False,
         skip_linear_probe=False, skip_pr=False, eval_every=10, lp_epochs=5, lp_lr=0.1, lp_wd=0.0,
         spectral_loss_coeff=0.0, spectral_loss_warmup_epochs=0, neural_ev_layer="encoder.layer4.0.bn1",
         neural_data_dir="src/REVERSE_PRED_FINAL/majajhong_cache", seed=0,):
    args = {"imagenet_root": imagenet_root, "epochs": epochs, "batch_size": batch_size,
            "img_size": img_size, "tau": tau, "lr": lr, "wd": wd, "workers": workers, "accum_steps": accum_steps,
            "warmup_epochs": warmup_epochs, "amp": amp, "grad_clip": grad_clip, "limit_train": limit_train, "limit_val": limit_val,
            "log_every": log_every, "save_dir": save_dir, "skip_knn": skip_knn, "skip_alpha": skip_alpha, "skip_neural_ev": skip_neural_ev,
            "skip_linear_probe": skip_linear_probe, "skip_pr": skip_pr, "eval_every": eval_every, "lp_epochs": lp_epochs, "lp_lr": lp_lr,
            "lp_wd": lp_wd, "spectral_loss_coeff": spectral_loss_coeff, "spectral_loss_warmup_epochs": spectral_loss_warmup_epochs,
            "neural_ev_layer": neural_ev_layer, "neural_data_dir": neural_data_dir, "seed": seed}
    set_seed(seed)
    
        # save dir
    start_ts = time.strftime("%Y%m%d-%H%M%S")
    save_dir = save_dir.strip()
    if save_dir:
        save_dir = os.path.join(save_dir, f"start_{start_ts}")
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, "ckpts", "simclr"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "logs"), exist_ok=True)
    else:
        os.makedirs("ckpts/simclr", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
    epochs_completed = 0
    best_ssl_val = float("inf")
    best_linear_probe = 0.0

    # device preference
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print("Using device:", device)

    pin_memory = (device == "cuda")
    amp_enabled = (amp and device == "cuda")
    scaler = GradScaler("cuda") if amp_enabled else None



    print(f"beta (spectral loss coeff): {spectral_loss_coeff}")
    print(f"tau={tau} bs={batch_size} lr={lr} wd={wd} warmup={warmup_epochs} epochs={epochs}")
    print(f"workers={workers} amp={amp_enabled} grad_clip={grad_clip}")
    print(f"eval_every={eval_every} (knn/linear probe/pr), seed={seed}")

    # --------------------------
    # Loaders
    # --------------------------
    ssl_train_dl = build_ssl_train_loader(
        root=imagenet_root,
        batch_size=batch_size,
        workers=workers,
        img_size=img_size,
        pin_memory=pin_memory,
        limit_train=limit_train,
    )

    ssl_val_dl = build_ssl_val_loader(
        root=imagenet_root,
        batch_size=batch_size,
        workers=workers,
        img_size=img_size,
        pin_memory=pin_memory,
        limit_val=limit_val,
    )

    # clean eval loaders for kNN + linear probe
    eval_tr_dl, eval_va_dl = build_eval_loaders(
        root=imagenet_root,
        batch_size=batch_size,
        workers=workers,
        img_size=img_size,
        pin_memory=pin_memory,
        limit_train=None,
        limit_val=None,
    )

    base_ds = eval_tr_dl.dataset.dataset if hasattr(eval_tr_dl.dataset, "dataset") else eval_tr_dl.dataset
    num_classes = len(base_ds.classes)

    # --------------------------
    # Model + hooks
    # --------------------------
    model = SimCLR(out_dim=128).to(device)

    activationclass = ModelActivations(model, layers=[neural_ev_layer])
    activationclass.register_hooks()

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)

    # logging
    log_path = os.path.join(save_dir, "logs", "simclr_baseline.csv") if save_dir else "logs/simclr_baseline.csv"
    header = [
        "ts", "epoch", "lr", "tau", "batch_size", "img_size",
        "train_loss", "ssl_val_loss",
        "knn_top1", "linear_probe_top1",
        "PR_z", "lam1_z", "lam_min_z",
        "alpha", "beta",
        "BPI", "F_EV", "R_EV",
        "device", "seed", "spec_loss_warmup_epochs"
    ]
    if not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(header)

    # --------------------------
    # Train
    # --------------------------
    model.train()
    steps_per_epoch = len(ssl_train_dl)

    for epoch in range(epochs_completed, epochs):
        epoch_loss = 0.0
        n_steps = 0

        optimizer.zero_grad(set_to_none=True)

        for it, (q, k) in enumerate(ssl_train_dl):
            q = q.to(device, non_blocking=True).contiguous()
            k = k.to(device, non_blocking=True).contiguous()

            lr_now = cosine_lr_with_warmup(
                optimizer,
                base_lr=lr,
                epoch=epoch,
                step_in_epoch=it,
                steps_per_epoch=steps_per_epoch,
                warmup_epochs=warmup_epochs,
                total_epochs=epochs,
            )

            if amp_enabled:
                with autocast("cuda"):
                    z1 = model(q)
                    z2 = model(k)
                    # l1 = info_nce(z1, z2, tau=args.tau) / args.accum_steps
                    l1 = info_nce(z1, z2, tau=tau)


                    if spectral_loss_coeff != 0.0 and epoch >= int(spectral_loss_warmup_epochs):
                        acts = activationclass.activations[neural_ev_layer]
                        l2, alpha = spectral_loss(acts, device)

                        assert acts.requires_grad
                        assert l2.requires_grad

                    else:
                        l2 = torch.tensor(0.0, device=device)
                        alpha = torch.tensor(0.0, device=device)

                    loss = l1 + spectral_loss_coeff * l2
                    loss = loss / accum_steps

                scaler.scale(loss).backward()

                if (it + 1) % accum_steps == 0:
                    if grad_clip and grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

            else:
                z1 = model(q)
                z2 = model(k)
                # l1 = info_nce(z1, z2, tau=args.tau) / args.accum_steps
                l1 = info_nce(z1, z2, tau=tau)

                if spectral_loss_coeff != 0.0 and epoch >= int(spectral_loss_warmup_epochs):
                    acts = activationclass.activations[neural_ev_layer]
                    l2, alpha = spectral_loss(acts, device)

                    assert acts.requires_grad
                    assert l2.requires_grad

                else:
                    l2 = torch.tensor(0.0, device=device)
                    alpha = torch.tensor(0.0, device=device)

                loss = l1 + spectral_loss_coeff * l2
                loss = loss / accum_steps
                loss.backward()

                if (it + 1) % accum_steps == 0:
                    if grad_clip and grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            epoch_loss += loss.item() * accum_steps
            n_steps += 1

            if (it + 1) % log_every == 0:
                print(f"epoch {epoch+1} iter {it+1}/{steps_per_epoch} lr {lr_now:.5f} loss {epoch_loss/max(1,n_steps):.4f} | alpha {alpha:.3f}")

        train_avg = epoch_loss / max(1, n_steps)
        print(f"--- Epoch {epoch+1} done | avg train loss {train_avg:.4f} ---")

        # --------------------------
        # SSL val loss (proper)
        # --------------------------
        ssl_val_avg, val_alpha = ssl_val_infonce_and_alpha(
            model,
            ssl_val_dl,
            tau=tau,
            device=device,
            activationclass=activationclass,
            alpha_layer=neural_ev_layer,
            compute_alpha=(not skip_alpha),
        )
        print(f"epoch {epoch+1} | ssl val InfoNCE {ssl_val_avg:.4f} | alpha {val_alpha:.3f}")
        
        # --------------------------
        # Alpha metric (optional)
        # --------------------------
        # if args.skip_alpha:
        #     val_alpha = 0.0
        # # else:
        # #     with torch.no_grad():
        # #         q, _ = next(iter(ssl_val_dl))
        # #         q = q.to(device, non_blocking=True).contiguous()
        # #         _ = model(q)
        # #         val_alpha = float(just_alpha(activationclass.activations[args.neural_ev_layer], device=device).cpu().item())
        # print(f"epoch {epoch+1} | alpha {val_alpha:.3f}")

        # --------------------------
        # kNN + linear probe + PR (cadenced)
        # --------------------------
        do_eval = ((epoch + 1) % eval_every == 0) or (epoch == 0)

        knn_acc = 0.0
        lp_acc = 0.0
        pr_z = 0.0
        lam1_z = 0.0
        lammin_z = 0.0
        bpi = 0.0
        f_ev = 0.0
        r_ev = 0.0

        if do_eval:
            if not skip_knn:
                knn_acc = knn_top1(model.encoder, eval_tr_dl, eval_va_dl, device=device, k=1)
                print(f"epoch {epoch+1} | kNN(1) top1 {knn_acc:.2f}%")

            if not skip_linear_probe:
                lp_acc = linear_probe_top1(
                    model.encoder, eval_tr_dl, eval_va_dl, num_classes=num_classes,
                    device=device, epochs=lp_epochs, lr=lp_lr, wd=lp_wd
                )
                print(f"epoch {epoch+1} | linear probe top1 {lp_acc:.2f}%")

            if not skip_pr:
                pr_z, lam1_z, lammin_z = spectrum_pr(
                    model, ssl_train_dl, device=device, batches=2, max_per_batch=64, use="z"
                )
                print(f"epoch {epoch+1} | PR(z) {pr_z:.1f} | lam1 {lam1_z:.3g} | lam_min {lammin_z:.3g}")
            
            if not skip_neural_ev:
                neural_activations = np.load(os.path.join(neural_data_dir, "neural_activations.npy"))

                model_activations, stimulus_ids = extract_model_activations_from_cache(
                        model=model,
                        cache_dir=neural_data_dir,#"REVERSE_PRED_FINAL/majajhong_cache"
                        layer_name=neural_ev_layer,  # Auto-detect
                        batch_size=32
                    )
                f_ev, r_ev = forward_ev(model_activations, neural_activations), reverse_ev(model_activations, neural_activations)
                bpi = 2*f_ev*r_ev/(f_ev + r_ev + 1e-12)
                
                # ev_dict = F_R_EV(
                # model,
                # activation_layer=neural_ev_layer,
                # neural_data_dir=neural_data_dir,
                # alpha=0.5,
                # transforms=three_channel_transform,
                # reliability_threshold=0.7,
                # batch_size=4,)
                # bpi, f_ev, r_ev = ev_dict["BPI"], ev_dict["F_EV"], ev_dict["R_EV"]

                print(f"epoch {epoch+1} | BPI {bpi:.3f} | F_EV {f_ev:.3f} | R_EV {r_ev:.3f}")
            # else:
            #     ev_dict = {"BPI": 0.0, "F_EV": 0.0, "R_EV": 0.0}
            #     bpi, f_ev, r_ev = ev_dict["BPI"], ev_dict["F_EV"], ev_dict["R_EV"]


        # # --------------------------
        # # Neural EV (optional, expensive)
        # # --------------------------
        # if args.skip_neural_ev:
        #     ev_dict = {"BPI": 0.0, "F_EV": 0.0, "R_EV": 0.0}
        # else:
        #     ev_dict = F_R_EV(
        #         model,
        #         activation_layer=args.neural_ev_layer,
        #         neural_data_dir=args.neural_data_dir,
        #         alpha=0.5,
        #         transforms=three_channel_transform,
        #         reliability_threshold=0.7,
        #         batch_size=4,
        #     )

        # bpi, f_ev, r_ev = ev_dict["BPI"], ev_dict["F_EV"], ev_dict["R_EV"]
        # if not args.skip_neural_ev:
        #     print(f"epoch {epoch+1} | BPI {bpi:.3f} | F_EV {f_ev:.3f} | R_EV {r_ev:.3f}")

        # --------------------------
        # Checkpointing
        # --------------------------
        ts = int(time.time())
        ckpt = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "opt": optimizer.state_dict(),
            "args": args,
            "best_ssl_val": best_ssl_val,
            "best_linear_probe": best_linear_probe,
        }

        ckpt_dir = os.path.join(save_dir, "ckpts", "simclr") if save_dir else "ckpts/simclr"
        last_path = os.path.join(ckpt_dir, "last.pt")
        torch.save(ckpt, last_path)

        if ssl_val_avg < best_ssl_val:
            best_ssl_val = ssl_val_avg
            best_path = os.path.join(ckpt_dir, "best_ssl_val.pt")
            torch.save(ckpt, best_path)
            print(f"epoch {epoch+1} | new best ssl val {best_ssl_val:.4f} -> saved best_ssl_val.pt")

        if lp_acc > best_linear_probe:
            best_linear_probe = lp_acc
            best_lp_path = os.path.join(ckpt_dir, "best_linear_probe.pt")
            torch.save(ckpt, best_lp_path)
            print(f"epoch {epoch+1} | new best linear probe {best_linear_probe:.2f}% -> saved best_linear_probe.pt")

        # --------------------------
        # Log row
        # --------------------------
        row = [
            ts, epoch + 1, optimizer.param_groups[0]["lr"], tau, batch_size, img_size,
            train_avg, ssl_val_avg,
            knn_acc, lp_acc,
            pr_z, lam1_z, lammin_z,
            val_alpha, spectral_loss_coeff,
            bpi, f_ev, r_ev,
            device, seed, spectral_loss_warmup_epochs
        ]
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(row)
    return bpi


if __name__ == "__main__":
    main()
