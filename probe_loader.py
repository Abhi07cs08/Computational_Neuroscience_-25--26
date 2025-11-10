# scripts/probe_loader.py
import os, torch, time
from src.datamod.imagenet_ssl import build_imagenet_loaders

root = os.path.expanduser("~/datasets/imagenet1k_small")
dl = build_imagenet_loaders(root, batch_size=32, workers=0, img_size=128, pin_memory=False, limit_train=512)

print("Iterating...")
t0 = time.time()
it = iter(dl)
q,k = next(it)   # force-load first batch
print("First batch:", q.shape, k.shape, "time:", time.time()-t0, "s")
