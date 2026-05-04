"""
src/data/augmentation.py
Data augmentation pipelines for training and validation.
Uses torchvision transforms with optional Albumentations integration.
"""

from torchvision import transforms


# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Aggressive augmentation pipeline for training.
    Techniques used:
      - Random horizontal flip                  → invariance to orientation
      - Random rotation (±15°)                  → rotational variation
      - Color jitter                            → lighting variation
      - Random resized crop                     → scale/position invariance
      - Gaussian blur                           → robustness to focus issues
      - Random erasing                          → occlusion robustness
      - Normalize (ImageNet stats)              → pre-trained model compatibility
    """
    return transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05,
        ),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
        ], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
    ])


def get_val_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Deterministic pipeline for validation and test sets.
    No augmentation — only resize, center-crop, and normalize.
    """
    return transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_inference_transforms(image_size: int = 224) -> transforms.Compose:
    """
    Minimal pipeline for single-image real-time inference.
    Same as val but separated for clarity.
    """
    return get_val_transforms(image_size)


def denormalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Reverse ImageNet normalization for visualization."""
    import torch
    mean = torch.tensor(mean).view(3, 1, 1)
    std  = torch.tensor(std).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)
