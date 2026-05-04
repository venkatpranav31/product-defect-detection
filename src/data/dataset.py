"""
src/data/dataset.py
Custom PyTorch Dataset for Product Defect Detection.
Supports train/val/test splits from a structured folder.
"""

import os
from pathlib import Path
from typing import Optional, Tuple, List

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import numpy as np


CLASS_MAP = {
    "non_defective": 0,
    "defective":     1,
}


class DefectDataset(Dataset):
    """
    Dataset loader for defect classification.

    Expected folder structure:
        root/
          train/
            defective/      *.jpg | *.png
            non_defective/  *.jpg | *.png
          val/   ...
          test/  ...
    """

    VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
    ):
        self.root_dir = Path(root_dir) / split
        self.transform = transform
        self.split = split
        self.samples: List[Tuple[Path, int]] = []

        self._load_samples()

    # ------------------------------------------------------------------
    def _load_samples(self):
        for class_name, label in CLASS_MAP.items():
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                raise FileNotFoundError(
                    f"[DefectDataset] Class folder not found: {class_dir}"
                )
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in self.VALID_EXTENSIONS:
                    self.samples.append((img_path, label))

        if len(self.samples) == 0:
            raise ValueError(f"[DefectDataset] No images found in {self.root_dir}")

        print(
            f"[DefectDataset] Loaded {len(self.samples)} images "
            f"({self.split}) — "
            f"Defective: {self._count(1)}, Non-Defective: {self._count(0)}"
        )

    def _count(self, label: int) -> int:
        return sum(1 for _, l in self.samples if l == label)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

    # ------------------------------------------------------------------
    def get_class_weights(self) -> torch.Tensor:
        """Returns inverse-frequency class weights for loss weighting."""
        counts = np.array([self._count(0), self._count(1)], dtype=np.float32)
        weights = 1.0 / (counts + 1e-6)
        weights /= weights.sum()
        return torch.tensor(weights, dtype=torch.float32)

    def get_sampler(self) -> WeightedRandomSampler:
        """Returns a WeightedRandomSampler to handle class imbalance."""
        labels = np.array([label for _, label in self.samples])
        class_counts = np.bincount(labels, minlength=2)
        class_weights = 1.0 / (class_counts + 1e-6)
        sample_weights = class_weights[labels]
        return WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.float32),
            num_samples=len(sample_weights),
            replacement=True,
        )


# ------------------------------------------------------------------
def build_dataloaders(
    data_dir: str,
    train_transform: transforms.Compose,
    val_transform: transforms.Compose,
    batch_size: int = 32,
    num_workers: int = 4,
    use_sampler: bool = True,
) -> dict:
    """Build train/val/test DataLoaders."""
    datasets = {
        "train": DefectDataset(data_dir, "train", train_transform),
        "val":   DefectDataset(data_dir, "val",   val_transform),
        "test":  DefectDataset(data_dir, "test",  val_transform),
    }

    sampler = datasets["train"].get_sampler() if use_sampler else None

    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            sampler=sampler,
            shuffle=(sampler is None),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
    }
    return loaders, datasets
