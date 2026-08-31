from src.utils.construct_df import construct_df
import torch
from pathlib import Path
import numpy as np
from src.utils.post_training import extract_model_brainscore_acts

df = construct_df()
df = df[((df["version"]=="fixed alpha_loss_04042026") & (df["epochs"]>=198) & (df["recomputed_alpha"]<=1.6) & (df["tau"]==0.2) & (df["target_alpha"]<=2))| ((df["spectral_loss_coeff"]==0)& (df["tau"]==0.2))]
for cp in df["ckpt_path"].tolist():
    print(f"Processing checkpoint: {cp}")
    root_cp = Path(cp).parent.parent.parent
    print(f"Root checkpoint path: {root_cp}")
    if not root_cp.exists():
        print(f"Missing root checkpoint path: {root_cp}")
    if not (root_cp / "brain_score_acts.pt").exists():
        print(f"Missing brain score acts for: {root_cp}")
        acts = extract_model_brainscore_acts(cp, "/home/kostouso/CompNeuro/Computational_Neuroscience_-25--26/src/latest_neural_data/majajhong_cache/")
        torch.save(acts, root_cp / "brain_score_acts.pt")
        print(f"Saved brain score acts for: {root_cp}")
    if not (root_cp / "brain_score_acts_svd.pt").exists():
        print(f"Missing brain score acts SVD for: {root_cp}")
        acts = torch.load(root_cp / "brain_score_acts.pt", weights_only=False)
        Uh, Sh, Vh = np.linalg.svd(acts, full_matrices=False)
        torch.save((Uh, Sh, Vh), root_cp / "brain_score_acts_svd.pt")
        print(f"Saved brain score acts SVD for: {root_cp}")
print("SVD computation completed for all checkpoints.")