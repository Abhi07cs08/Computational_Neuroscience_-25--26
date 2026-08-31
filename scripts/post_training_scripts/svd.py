from src.utils.construct_df import construct_df
import torch
from pathlib import Path
import numpy as np
from src.utils.post_training import extract_model_brainscore_acts

df = construct_df()
for cp in df["checkpoint_path"].tolist():
    print(f"Processing checkpoint: {cp}")
    root_cp = Path(cp).parent.parent
    print(f"Root checkpoint path: {root_cp}")
    if not root_cp.exists():
        print(f"Missing root checkpoint path: {root_cp}")
    if not (root_cp / "brain_score_acts.pt").exists():
        print(f"Missing brain score acts for: {root_cp}")
        acts = extract_model_brainscore_acts(cp, "/home/kostouso/CompNeuro/Computational_Neuroscience_-25--26/src/latest_neural_data/majajhong_cache/stimulus_ids.npy")
        torch.save(acts, root_cp / "brain_score_acts.pt")
    else:
        acts = torch.load(root_cp / "brain_score_acts.pt")
    if not (root_cp / "brain_score_acts_svd.pt").exists():
        print(f"Missing brain score acts SVD for: {root_cp}")
        Uh, Sh, Vh = np.linalg.svd(acts, full_matrices=False)
        torch.save((Uh, Sh, Vh), root_cp / "brain_score_acts_svd.pt")
    else:
        Uh, Sh, Vh = torch.load(root_cp / "brain_score_acts_svd.pt")