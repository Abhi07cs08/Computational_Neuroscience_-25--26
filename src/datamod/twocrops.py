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
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([cj], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=int(0.1 * img_size) | 1, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

def simclr_transform_cifar(img_size=32):
    cj = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)

    # CIFAR-10 mean/std (common)
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0)),  # less destructive than 0.08
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([cj], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=max(3, (int(0.1 * img_size) | 1)),
                                     sigma=(0.1, 1.0))],
            p=0.5
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def ssl_deterministic_transform(img_size=224):
    # deterministic "view" for SSL diagnostics
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

def eval_train_transform(img_size=224, deterministic=True):
    if deterministic:
        return transforms.Compose([
            transforms.Resize(int(img_size * 1.14)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
    # optional, if you want "train aug" eval later
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

def eval_val_transform(img_size=224):
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

def eval_train_transform_cifar(img_size=32, deterministic=True):
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    if deterministic:
        # True eval: no spatial distortion
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    # Optional: evaluate with light train-style augmentation
    return transforms.Compose([
        transforms.RandomCrop(img_size, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

def eval_val_transform_cifar(img_size=32):
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

