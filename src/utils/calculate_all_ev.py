from pathlib import Path
import sys
from typing import Union
import os

import matplotlib.pyplot as plt
from src.latest_neural_data.ev_helper import forward_ev, reverse_ev
from src.latest_neural_data.model_acts import extract_model_activations_from_cache
from src.eval.linear_probe import linear_probe_standalone, linear_probe_adversarial_robustness_standalone
import pandas as pd
from src.losses.spectral_loss import just_alpha_imgnet_standalone, just_alpha_brainscore_standalone
from reverse_pred.monkey_to_model import compute_monkey_to_model
from reverse_pred.model_to_monkey import compute_model_to_monkey
from src.utils.post_training import extract_model_brainscore_acts_with_neural
import numpy as np
import argparse




def fr_ev(ckpt_path, folder=None, neural_data_dir="src/latest_neural_data/majajhong_cache/"):
    # model_acts, neural_acts = extract_model_brainscore_acts_with_neural(ckpt_path, neural_data_dir="src/latest_neural_data/majajhong_cache" )
    model_acts, neural_acts = extract_model_brainscore_acts_with_neural(ckpt_path, neural_data_dir=neural_data_dir )

    if folder is None:
        folder = ckpt_path.split("/")[:-1]
    if not os.path.exists(os.path.join("/".join(folder), "reverse_ev.npy")):
        compute_monkey_to_model(
            model_features=model_acts,
            rates =neural_acts,
            out_dir=folder,
            max_n=None,
            reps=20,
            out_name="reverse_ev.npy",)
        print(f"Saved reverse EV to {os.path.join('/'.join(folder), 'reverse_ev.npy')}")
    else:
        print(f"Reverse EV already exists at {os.path.join('/'.join(folder), 'reverse_ev.npy')}")
    if not os.path.exists(os.path.join("/".join(folder), "forward_ev.npy")):
        compute_model_to_monkey(
            rates_predictor=neural_acts,
            rates_predicted=model_acts,
            out_dir=folder,
            max_n=None,
            reps=20,
            out_name="forward_ev.npy",)
        print(f"Saved forward EV to {os.path.join('/'.join(folder), 'forward_ev.npy')}")
    else:
        print(f"Forward EV already exists at {os.path.join('/'.join(folder), 'forward_ev.npy')}")
    r_ev_path = os.path.join("/".join(folder), "reverse_ev.npy")
    f_ev_path = os.path.join("/".join(folder), "forward_ev.npy")
    return r_ev_path, f_ev_path

def calculate_ev_mean(ev_path):
    ev_values = np.load(ev_path)
    mean_ev = np.mean(ev_values)
    return mean_ev

def add_frev_to_csv(csv_path, fev_value, rev_value):
    df = pd.read_csv(csv_path)
    df["Forward_EV_brainscore_recomputed"] = [fev_value] * len(df)
    df["Reverse_EV_brainscore_recomputed"] = [rev_value] * len(df)
    df.to_csv(csv_path, index=False)

# scratch_path = Path("/mnt/d/CompNeuroTrials/")




def parse_args():
    parser = argparse.ArgumentParser(description="Calculate EV for a given checkpoint.")
    parser.add_argument("--mega_folder", type=str, default = "/scratch/kostouso/CompNeuro/Computational_Neuroscience_-25--26/", help="Path to the mega folder containing checkpoints.")
    parser.add_argument("--neural_data_dir", type=str, default="/home/kostouso/CompNeuro/Computational_Neuroscience_-25--26/src/latest_neural_data/majajhong_cache/", help="Path to the neural data directory")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    scratch_path = Path(args.mega_folder)
    model_paths = []
    for root, dirs, files in os.walk(scratch_path):
        if os.path.basename(root) == "ckpts":
            try:
                folder = os.path.dirname(root)
                ckpt_path = os.path.join(root, "simclr/last.pt")
                csv_path = os.path.join(folder, "logs/simclr_baseline.csv")
                ev_path = os.path.join(folder, "neural_predictivity")
                if not os.path.exists(csv_path) or not os.path.exists(ckpt_path):
                    print(f"Missing files in {root}, skipping.")
                    continue
                if not os.path.exists(ev_path):
                    os.makedirs(ev_path)
                r_ev_path, f_ev_path = fr_ev(ckpt_path, folder = ev_path, neural_data_dir=args.neural_data_dir)
                r_ev_mean = calculate_ev_mean(r_ev_path)
                f_ev_mean = calculate_ev_mean(f_ev_path)
                add_frev_to_csv(csv_path, f_ev_mean, r_ev_mean)
            except Exception as e:
                print(f"Error processing {root}: {e}")
