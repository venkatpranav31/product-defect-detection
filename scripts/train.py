"""
scripts/train.py
Entry point for model training.

Usage:
    python scripts/train.py --config config/config.yaml
    python scripts/train.py --data_dir data/processed --epochs 30 --batch_size 32
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# Make src importable from root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.augmentation import get_train_transforms, get_val_transforms
from src.data.dataset import build_dataloaders
from src.models.resnet50 import build_model
from src.training.trainer import Trainer


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser(description="Train Defect Detection Model")
    parser.add_argument("--config",      type=str, default="config/config.yaml")
    parser.add_argument("--data_dir",    type=str, default=None)
    parser.add_argument("--output_dir",  type=str, default=None)
    parser.add_argument("--log_dir",     type=str, default=None)
    parser.add_argument("--epochs",      type=int, default=None)
    parser.add_argument("--batch_size",  type=int, default=None)
    parser.add_argument("--lr",          type=float, default=None)
    parser.add_argument("--freeze",      type=int, default=None, help="Layers to freeze (0-4)")
    parser.add_argument("--no_amp",      action="store_true", help="Disable mixed precision")
    parser.add_argument("--seed",        type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Load Config ──────────────────────────────────────────────────
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # CLI overrides
    if args.data_dir:   cfg["paths"]["data_dir"]    = args.data_dir
    if args.output_dir: cfg["paths"]["output_dir"]  = args.output_dir
    if args.log_dir:    cfg["paths"]["log_dir"]      = args.log_dir
    if args.epochs:     cfg["training"]["epochs"]    = args.epochs
    if args.batch_size: cfg["training"]["batch_size"] = args.batch_size
    if args.lr:         cfg["training"]["learning_rate"] = args.lr
    if args.freeze:     cfg["model"]["freeze_layers"] = args.freeze
    if args.no_amp:     cfg["training"]["mixed_precision"] = False

    train_cfg   = cfg["training"]
    model_cfg   = cfg["model"]
    path_cfg    = cfg["paths"]
    data_cfg    = cfg["data"]

    set_seed(args.seed)

    print("\n" + "="*60)
    print("  Product Defect Detection — Training")
    print("="*60)

    # ── Transforms ───────────────────────────────────────────────────
    train_tf = get_train_transforms(data_cfg["image_size"])
    val_tf   = get_val_transforms(data_cfg["image_size"])

    # ── Data ─────────────────────────────────────────────────────────
    loaders, datasets = build_dataloaders(
        data_dir=path_cfg["data_dir"],
        train_transform=train_tf,
        val_transform=val_tf,
        batch_size=train_cfg["batch_size"],
        num_workers=data_cfg["num_workers"],
        use_sampler=True,
    )

    # ── Model ─────────────────────────────────────────────────────────
    model = build_model(
        num_classes=model_cfg["num_classes"],
        pretrained=model_cfg["pretrained"],
        freeze_layers=model_cfg["freeze_layers"],
        dropout_rate=model_cfg["dropout_rate"],
    )

    # Optional: pass class weights to loss function
    class_weights = datasets["train"].get_class_weights()
    train_cfg["class_weights"] = class_weights

    # ── Trainer ──────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        loaders=loaders,
        config=train_cfg,
        output_dir=path_cfg["output_dir"],
        log_dir=path_cfg["log_dir"],
    )

    history = trainer.train()

    print("\n✅ Training finished.")
    print(f"   Best Val Accuracy: {trainer.best_val_acc:.4f}")
    print(f"   Checkpoint saved to: {path_cfg['output_dir']}/best_model.pth")
    print("\nNext step — evaluate on test set:")
    print(f"   python scripts/evaluate.py --model {path_cfg['output_dir']}/best_model.pth")


if __name__ == "__main__":
    main()
