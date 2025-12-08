#!/usr/bin/env python3
import argparse
import os
import random
import shutil
from typing import List

"""
create_train_val_split.py

Shuffle images from a folder (recursively) and split them into train/ and val/ folders,
preserving relative subfolder structure.

Usage:
    python create_train_val_split.py /path/to/images --val-size 0.2 --seed 42 --move

By default the script will copy files. Use --move to move files instead.
"""

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tif', '.tiff'}


def list_image_files(folder: str, exts: set) -> List[str]:
    files = []
    for root, _, filenames in os.walk(folder):
        for name in filenames:
            _, e = os.path.splitext(name)
            if e.lower() in exts:
                files.append(os.path.join(root, name))
    return files


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def split_and_move_or_copy(src_folder: str, train_folder: str, val_folder: str,
                           val_size: float, seed: int, move_files: bool):
    src_abs = os.path.abspath(src_folder)
    train_abs = os.path.abspath(train_folder)
    val_abs = os.path.abspath(val_folder)

    all_files = list_image_files(src_folder, IMAGE_EXTS)

    # exclude any files that are inside the train/ or val/ target folders (if they are within src)
    def is_under(path: str, parent_abs: str) -> bool:
        return os.path.commonpath([os.path.abspath(path), parent_abs]) == parent_abs

    all_files = [f for f in all_files if not (is_under(f, train_abs) or is_under(f, val_abs))]

    rng = random.Random(seed)
    rng.shuffle(all_files)

    n_total = len(all_files)
    n_val = int(round(n_total * val_size))
    val_files = all_files[:n_val]
    train_files = all_files[n_val:]

    # create base dirs
    ensure_dir(train_folder)
    ensure_dir(val_folder)

    op = shutil.move if move_files else shutil.copy2

    # helper to copy/move while preserving relative subpath
    def process_file(src_path: str, dest_base: str):
        rel = os.path.relpath(src_path, src_abs)
        dst = os.path.join(dest_base, rel)
        dst_dir = os.path.dirname(dst)
        ensure_dir(dst_dir)
        op(src_path, dst)

    for src in train_files:
        process_file(src, train_abs)
    for src in val_files:
        process_file(src, val_abs)

    print(f"Total images: {n_total}")
    print(f"Train: {len(train_files)} -> {train_abs}")
    print(f"Val:   {len(val_files)} -> {val_abs}")
    print("Operation:", "moved" if move_files else "copied")


def main():
    p = argparse.ArgumentParser(description="Shuffle images (recursively) and create train/val split")
    p.add_argument("--src", help="Source folder containing images (will be searched recursively)")
    p.add_argument("--train-dir", help="Output train folder (default: <src>/train)", default=None)
    p.add_argument("--val-dir", help="Output val folder (default: <src>/val)", default=None)
    p.add_argument("--val-size", type=float, help="Fraction for validation set (default: 0.2)", default=0.2)
    p.add_argument("--seed", type=int, help="Random seed (default: 0)", default=0)
    p.add_argument("--move", action="store_true", help="Move files instead of copying them")
    args = p.parse_args()

    src = os.path.abspath(args.src)
    if not os.path.isdir(src):
        raise SystemExit(f"Source folder does not exist: {src}")

    train_dir = os.path.abspath(args.train_dir) if args.train_dir else os.path.join(src, "train")
    val_dir = os.path.abspath(args.val_dir) if args.val_dir else os.path.join(src, "val")

    # Prevent accidental overlap: don't allow train/val to be the same as src
    if os.path.abspath(train_dir) == src or os.path.abspath(val_dir) == src:
        raise SystemExit("train/val folders must be different from the source folder")

    split_and_move_or_copy(src, train_dir, val_dir, args.val_size, args.seed, args.move)


if __name__ == "__main__":
    main()