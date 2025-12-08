import os
import shutil
import random
import argparse

random.seed(42)

parser = argparse.ArgumentParser(description="Split dataset into train and val sets")
parser.add_argument("--src_root", type=str, default="imagenet1k", help="Source root directory of the dataset")
parser.add_argument("--dst_train", type=str, default="train", help="Destination directory for training set")
parser.add_argument("--dst_val", type=str, default="val", help="Destination directory for validation set")
parser.add_argument("--train_ratio", type=float, default=0.8, help="Ratio of training data")
args = parser.parse_args()


src_root = args.src_root
dst_train = args.dst_train
dst_val = args.dst_val
train_ratio = args.train_ratio

os.makedirs(dst_train, exist_ok=True)
os.makedirs(dst_val, exist_ok=True)

for class_name in os.listdir(src_root):
    class_path = os.path.join(src_root, class_name)
    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)
    random.shuffle(images)

    split_index = int(len(images) * train_ratio)
    train_images = images[:split_index]
    val_images = images[split_index:]

    train_class_dir = os.path.join(dst_train, class_name)
    val_class_dir = os.path.join(dst_val, class_name)
    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(val_class_dir, exist_ok=True)

    for img in train_images:
        shutil.copy(os.path.join(class_path, img), os.path.join(train_class_dir, img))

    for img in val_images:
        shutil.copy(os.path.join(class_path, img), os.path.join(val_class_dir, img))

print("Dataset split completed.")