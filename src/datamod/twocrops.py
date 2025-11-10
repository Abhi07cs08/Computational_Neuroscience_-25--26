from torchvision import transforms
from PIL import Image

class TwoCropsTransform:
    def __init__(self, base_t):
        self.base_t = base_t
    def __call__(self, x: Image.Image):
        q = self.base_t(x)
        k = self.base_t(x)
        return q, k

def simclr_transform(img_size=160):  # was 224
    cj = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([cj], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=int(0.1*img_size) | 1, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
