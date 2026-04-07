import os
from types import SimpleNamespace
import pandas as pd
import torch
from src.latest_neural_data.ev_helper import forward_ev, reverse_ev
from src.latest_neural_data.model_acts import extract_model_activations_from_cache
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from src.models.simclr_model import SimCLR
import torch.nn as nn
import torch.nn.functional as F
from src.datamod.imagenet_ssl import (
    build_ssl_train_loader,
    build_ssl_val_loader,
    build_eval_loaders,
)

def fetch_all_paths(root_dir):
    model_paths = []
    for root, dirs, files in os.walk(root_dir):
        if os.path.basename(root) == "ckpts":
            folder = os.path.dirname(root)
            model_path = os.path.join(root, "simclr/last.pt")
            csv_path = os.path.join(folder, "logs/simclr_baseline.csv")
            if os.path.exists(csv_path) and os.path.exists(model_path):
                model_paths.append((csv_path, model_path))
    return model_paths

def extract_stats(csv_path):
    data = pd.read_csv(csv_path)
    last_informative = {}
    for i in range(len(data)):
        values = dict(data.iloc[i])
        if values["F_EV"] != 0.0:
            last_informative = values
    return last_informative

def extract_ckpt_args(ckpt_path, as_args=False):
    ckpt = torch.load(ckpt_path, weights_only=False)
    args = ckpt["args"]
    if as_args:
        args = SimpleNamespace(**args)
    return args

def extract_val_dl_from_ckpt(ckpt_path, kwargs={}):
    args = extract_ckpt_args(ckpt_path, as_args=True)
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print("Using device:", device)

    pin_memory = (device == "cuda")
    for key, value in kwargs.items():
        setattr(args, key, value)
    eval_tr_dl, eval_va_dl = build_eval_loaders(
        root=args.imagenet_root,
        batch_size=args.batch_size,
        workers=args.workers,
        img_size=args.img_size,
        pin_memory=pin_memory,
        limit_train=None,
        limit_val=None,
        )
    return eval_tr_dl, eval_va_dl

def to_float(str):
    try:
        flt = float(str)
    except:
        flt = 0
    return flt

to_flt = np.vectorize(to_float)

def extract_model_weights(ckpt_path):
    ckpt = torch.load(ckpt_path, weights_only=False)
    model_weights = ckpt['model']
    return model_weights

def df_from_model_paths(model_paths):
    runs = []
    for i, (csv_path, ckpt_path) in enumerate(model_paths):
        args = extract_ckpt_args(ckpt_path)
        stats = extract_stats(csv_path)
        # if not "F_EV" in stats:
        #     continue
        for key, value in stats.items():
            if key not in args:
                args[key] = value
        args["ckpt_path"] = ckpt_path
        args["csv_path"] = csv_path
        runs.append(args)
    df = pd.DataFrame(runs)
    df["datetime"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    return df

def df_from_root_dir(root_dir):
    model_paths = fetch_all_paths(root_dir)
    df = df_from_model_paths(model_paths)
    return df

def extract_model_brainscore_acts(ckpt_path, stimulus_dir=None):
    args = extract_ckpt_args(ckpt_path, as_args=True)
    model = SimCLR()
    state_dict = extract_model_weights(ckpt_path)
    model.load_state_dict(state_dict)
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:        device = "cpu"
    model = model.to(device)

    if stimulus_dir is None:
        stimulus_dir = args.neural_data_dir


    model_activations, _, = extract_model_activations_from_cache(
            model=model,
            cache_dir=stimulus_dir,
            layer_name=args.neural_ev_layer,
            batch_size=args.batch_size,
        )
    return model_activations

def extract_model_brainscore_acts_with_neural(ckpt_path, neural_data_dir=None):
    args = extract_ckpt_args(ckpt_path, as_args=True)
    model = SimCLR()
    state_dict = extract_model_weights(ckpt_path)
    model.load_state_dict(state_dict)
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:        device = "cpu"
    model = model.to(device)

    if neural_data_dir is None:
        neural_data_dir = args.neural_data_dir


    model_activations, _, neural_activations = extract_model_activations_from_cache(
            model=model,
            cache_dir=neural_data_dir,
            layer_name=args.neural_ev_layer,
            batch_size=args.batch_size,
            return_neural_activations=True
        )
    return model_activations, neural_activations

def plot_ev_graph(r_ev, f_ev, bins_num=50, title="Histogram of Explained Variance"):
    bins = np.linspace(0, 100, bins_num)
    r_mean = np.nanmean(r_ev)
    f_mean = np.nanmean(f_ev)
    plt.figure(figsize=(8, 5))
    plt.hist(r_ev, bins=bins, color="steelblue", edgecolor="black", alpha=0.8, density=True, label="r_ev")
    plt.hist(f_ev, bins=bins, color="green", edgecolor="black", alpha=0.6, density=True, label="f_ev")
    plt.axvline(r_mean, color="navy", linestyle="--", linewidth=2, label=f"Reverse EV: {r_mean:.2f}")
    plt.axvline(f_mean, color="darkgreen", linestyle="--", linewidth=2, label=f"Forward EV: {f_mean:.2f}")
    plt.title(title)
    plt.xlabel("Explained Variance")
    plt.ylabel("Units")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_ev_from_df(df, neural_data_folder="/home/kostouso/CompNeuro/Computational_Neuroscience_-25--26/src/latest_neural_data/majajhong_cache/"):
    bins_num = 50
    # bins = np.linspace(0, 100, bins_num)
    neural_activations = np.load(os.path.join(neural_data_folder, "neural_activations.npy"))
    num = len(df.index)
    if isinstance(df, pd.Series):
        num = 1
        df = df.to_frame()
    for i in range(num):
        data = df.iloc[i]
        print(data["ckpt_path"])
        try:
            ckpt_path = data["ckpt_path"]
            model_weights = extract_model_weights(ckpt_path)
        except:
            continue
        model = SimCLR()
        model.load_state_dict(model_weights)
        layer = 'encoder.layer4.1'
    
        model_activations, stimulus_ids = extract_model_activations_from_cache(
            model=model,
            cache_dir=neural_data_folder,#"REVERSE_PRED_FINAL/majajhong_cache"
            layer_name=layer,
            batch_size=32
        )
        print(model_activations.shape)
        r_ev = reverse_ev(model_activations, neural_activations, full_ev_vector=True, unrevamped=False)
        f_ev = forward_ev(model_activations, neural_activations, full_ev_vector=True, unrevamped=False)

        X = model_activations - np.mean(model_activations, axis=0)
        pca = PCA(n_components=50)
        pca.fit(X)
        eigenvalues = pca.explained_variance_
        d_eff = (np.sum(eigenvalues) ** 2) / np.sum(eigenvalues ** 2)
        
        plot_ev_graph(r_ev, f_ev, bins_num=bins_num, title=f"Histogram of Explained Variance | alpha: {data['alpha']:.2f} | spectral loss coefficient: {data['spectral_loss_coeff']:.2f} | forward EV: {data['F_EV']:.2f} | reverse EV: {data['R_EV']:.2f} | tag: {data['tag']} | ED: {d_eff:.2f}")

        # plt.figure(figsize=(8, 5))
        # plt.hist(r_ev, bins=bins, color="steelblue", edgecolor="black", alpha=0.8, density=True)
        # plt.hist(f_ev, bins=bins, color="green", edgecolor="black", alpha=0.6, density=True)
        # plt.title(f"Histogram of Reverse EV | alpha: {data['alpha']:.2f} | spectral loss coefficient: {data['spectral_loss_coeff']:.2f} | forward EV: {data['F_EV']:.2f} | reverse EV: {data['R_EV']:.2f} | tag: {data['tag']} | ED: {d_eff:.2f}")
        # plt.xlabel("Explained Variance")
        # plt.ylabel("Units")
        # plt.grid(axis="y", alpha=0.3)
        # plt.tight_layout()
        # plt.show()

def plot_ev_ckpt(ckpt_path, neural_data_folder="/home/kostouso/CompNeuro/Computational_Neuroscience_-25--26/src/latest_neural_data/majajhong_cache/"):
    bins_num = 50
    bins = np.linspace(0, 100, bins_num)
    neural_activations = np.load(os.path.join(neural_data_folder, "neural_activations.npy"))
    print(ckpt_path)
    model_weights = extract_model_weights(ckpt_path)
    model = SimCLR()
    model.load_state_dict(model_weights)
    layer = 'encoder.layer4.1'

    args = extract_ckpt_args(ckpt_path, as_args=True)

    model_activations, stimulus_ids = extract_model_activations_from_cache(
        model=model,
        cache_dir=neural_data_folder,#"REVERSE_PRED_FINAL/majajhong_cache"
        layer_name=layer,
        batch_size=32
    )
    r_ev = reverse_ev(model_activations, neural_activations, full_ev_vector=True, unrevamped=True)
    f_ev = forward_ev(model_activations, neural_activations, full_ev_vector=True, unrevamped=True)
    
    X = model_activations - np.mean(model_activations, axis=0)
    pca = PCA(n_components=50)
    pca.fit(X)
    eigenvalues = pca.explained_variance_
    d_eff = (np.sum(eigenvalues) ** 2) / np.sum(eigenvalues ** 2)

    plot_ev_graph(r_ev, f_ev, bins_num=bins_num, title=f"Histogram of Explained Variance | alpha: {getattr(args.alpha, 'nan'):.2f} | spectral loss coefficient: {getattr(args.spectral_loss_coeff, 'nan'):.2f} | forward EV: {getattr(args.F_EV, 'nan'):.2f} | reverse EV: {getattr(args.R_EV, 'nan'):.2f} | tag: {getattr(args.tag, 'nan')} | ED: {d_eff:.2f}")

    # plt.figure(figsize=(8, 5))
    # plt.hist(r_ev, bins=bins, color="steelblue", edgecolor="black", alpha=0.8, density=True)
    # plt.hist(f_ev, bins=bins, color="green", edgecolor="black", alpha=0.6, density=True)
    # plt.title(f"Histogram of Reverse EV | alpha: {args.alpha:.2f} | spectral loss coefficient: {args.spectral_loss_coeff:.2f} | forward EV: {args.F_EV:.2f} | reverse EV: {args.R_EV:.2f} | tag: {args.tag} | ED: {d_eff:.2f}")
    # plt.xlabel("Explained Variance")
    # plt.ylabel("Units")
    # plt.grid(axis="y", alpha=0.3)
    # plt.tight_layout()
    # plt.show()