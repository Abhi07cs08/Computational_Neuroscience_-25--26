# src/datamod/twocrops.py
from torchvision import transforms
from PIL import Image


class TwoCropsTransform:
    def __init__(self, base_t):
        self.base_t = base_t

    def __call__(self, x: Image.Image):
        q = self.base_t(x)
        k = self.base_t(x)
        return q, k


def simclr_transform(img_size=224):
    cj = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)

    # GaussianBlur kernel must be odd and >= 3
    k = max(3, int(0.1 * img_size))
    if k % 2 == 0:
        k += 1

    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([cj], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=k, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def eval_train_transform(img_size=224):
    # For eval train split (kNN bank / linear probe train), keep light aug or none.
    # Light aug is ok, but DO NOT use SimCLR heavy aug.
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def eval_val_transform(img_size=224):
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
