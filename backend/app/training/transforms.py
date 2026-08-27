"""Train/eval transforms shared by the three datasets.

Augmentation stays mild on purpose: flips, light brightness/contrast jitter
and a gentle random crop. No strong distortion — it would bias perceived
age/gender (cf. CLAUDE.md training notes).
"""

from torchvision import transforms

from app.config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD

_normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

train_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.85, 1.0), ratio=(0.95, 1.05)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        _normalize,
    ]
)

eval_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        _normalize,
    ]
)
