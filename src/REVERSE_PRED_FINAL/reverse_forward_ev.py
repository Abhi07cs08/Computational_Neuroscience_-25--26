# Extract model activations from cached images (BrainScore-independent)
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize
import pickle
from model_acts import extract_model_activations_from_cache
from ev_helper import forward_ev, reverse_ev

# model = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
import torch
from simclr_model import SimCLR
from datetime import datetime
ckpt = torch.load(r"./start_20260113-140202/ckpts/simclr/best_linear_probe.pt", map_location="cpu", weights_only=False)
weights = ckpt['model']
model = SimCLR()
model.load_state_dict(weights, strict=False)
run_tag = datetime.now().strftime("%Y%m%d-%H%M%S")
model_name = model._get_name()

    # Extract features using only cached data
model_activations, stimulus_ids = extract_model_activations_from_cache(
        model=model,
        cache_dir="REVERSE_PRED_FINAL/majajhong_cache",
        layer_name="encoder.layer4",  # Auto-detect
        batch_size=32
    )

    # Now you have model_activations without needing BrainScore!
print(f"Extracted features: {model_activations.shape}")

np.save("REVERSE_PRED_FINAL/majajhong_cache/model_activations.npy", model_activations)

neural_activations = np.load("REVERSE_PRED_FINAL/majajhong_cache/neural_activations.npy")
model_activations = np.load("REVERSE_PRED_FINAL/majajhong_cache/model_activations.npy")

print(f"Reverse Explained Variance: {reverse_ev(model_activations, neural_activations):.2f}%")
print(f"Forward Explained Variance: {forward_ev(model_activations, neural_activations):.2f}%")
    
