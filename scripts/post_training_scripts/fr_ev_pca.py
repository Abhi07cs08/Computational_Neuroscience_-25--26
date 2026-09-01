from src.utils.construct_df import construct_df
import torch
from pathlib import Path
import numpy as np
from src.utils.post_training import extract_model_brainscore_acts
from src.latest_neural_data.ev_helper import forward_ev, reverse_ev
from src.utils.post_training import write_to_csv_from_ckpt, fetch_full_args_from_ckpt_path
from sklearn.decomposition import PCA



df = construct_df()
df = df[((df["version"]=="fixed alpha_loss_04042026") & (df["epochs"]>=198) & (df["recomputed_alpha"]<=1.6) & (df["tau"]==0.2) & (df["target_alpha"]<=2))| ((df["spectral_loss_coeff"]==0)& (df["tau"]==0.2))]
baseline_cp = "/scratch/kostouso/CompNeuro/Computational_Neuroscience_-25--26/Feb_7_launch/sk_logs_spec_loss_optuna_multi_test/start_20260311-162520_tuning_spectral_loss_coeff/ckpts/simclr/last.pt"
baseline_pca = torch.load(Path(baseline_cp).parent.parent.parent / "brain_score_acts_pca.pt", weights_only=False)
neural_acts = np.load("/home/kostouso/CompNeuro/Computational_Neuroscience_-25--26/src/latest_neural_data/majajhong_cache/neural_activations.npy")
for cp in df["ckpt_path"].tolist():
    root_dir = Path(cp).parent.parent.parent
    if not root_dir.exists():
        print(f"Missing root checkpoint path: {root_dir}")
        continue
    if not (root_dir / "brain_score_acts_pca.pt").exists():
        if not (root_dir / "brain_score_acts.pt").exists():
            print(f"Missing brain score acts for: {root_dir}")
            acts = extract_model_brainscore_acts(cp, "/home/kostouso/CompNeuro/Computational_Neuroscience_-25--26/src/latest_neural_data/majajhong_cache/")
            torch.save(acts, root_dir / "brain_score_acts.pt")
            print(f"Saved brain score acts for: {root_dir}")
        acts = torch.load(root_dir / "brain_score_acts.pt", weights_only=False)
        pca = PCA(n_components=168, svd_solver="randomized", random_state=42)
        pca.fit(acts)
        torch.save(pca, root_dir / "brain_score_acts_pca.pt")
    else:
        pca = torch.load(root_dir / "brain_score_acts_pca.pt", weights_only=False)
    if not (root_dir / "brain_score_acts_pca_r_ev.npy").exists() or not (root_dir / "brain_score_acts_pca_f_ev.npy").exists():
        print(f"Missing pca EVs for: {root_dir}")
        r_ev = reverse_ev(pca, neural_acts, full_ev_vector=True, unrevamped=True)
        f_ev = forward_ev(pca, neural_acts, full_ev_vector=True, unrevamped=True)
        np.save(Path(cp).parent.parent.parent / "brain_score_acts_pca_r_ev.npy", r_ev)
        np.save(Path(cp).parent.parent.parent / "brain_score_acts_pca_f_ev.npy", f_ev)
    r_ev_mean = np.nanmean(np.load(root_dir / "brain_score_acts_pca_r_ev.npy"))
    f_ev_mean = np.nanmean(np.load(root_dir / "brain_score_acts_pca_f_ev.npy"))
    print(f"Checkpoint: {cp}, Mean Reverse EV: {r_ev_mean}, Mean Forward EV: {f_ev_mean}")
    write_to_csv_from_ckpt(cp, {"pca_r_ev": r_ev_mean, "pca_f_ev": f_ev_mean})
    assert "pca_r_ev" in fetch_full_args_from_ckpt_path(cp).keys(), f"pca_r_ev not found in checkpoint args for {cp}"
            


