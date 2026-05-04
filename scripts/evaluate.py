"""
scripts/evaluate.py
Entry point for model evaluation on the test set.

Usage:
    python scripts/evaluate.py \
        --model outputs/models/best_model.pth \
        --data_dir data/processed \
        --output_dir outputs/reports
"""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.augmentation import get_val_transforms
from src.data.dataset import DefectDataset
from torch.utils.data import DataLoader
from src.models.resnet50 import load_model
from src.evaluation.evaluator import Evaluator


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Defect Detection Model")
    parser.add_argument("--model",      type=str, required=True, help="Path to .pth checkpoint")
    parser.add_argument("--config",     type=str, default="config/config.yaml")
    parser.add_argument("--data_dir",   type=str, default=None)
    parser.add_argument("--split",      type=str, default="test", choices=["test", "val", "train"])
    parser.add_argument("--output_dir", type=str, default="outputs/reports")
    parser.add_argument("--batch_size", type=int, default=32)
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data_dir  = args.data_dir or cfg["paths"]["data_dir"]
    img_size  = cfg["data"]["image_size"]
    n_workers = cfg["data"]["num_workers"]

    print(f"\n  Evaluating on '{args.split}' split...")

    transform = get_val_transforms(img_size)
    dataset   = DefectDataset(data_dir, args.split, transform)
    loader    = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=True,
    )

    model = load_model(args.model, num_classes=cfg["model"]["num_classes"])

    evaluator = Evaluator(model, loader, output_dir=args.output_dir)
    metrics   = evaluator.evaluate()

    print(f"\n  Reports saved to: {args.output_dir}")
    return metrics


if __name__ == "__main__":
    main()
