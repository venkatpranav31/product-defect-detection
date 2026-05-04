"""
scripts/preprocess.py
Split raw data into train/val/test folders.

Usage:
    python scripts/preprocess.py \
        --input data/raw \
        --output data/processed \
        --split 0.8 0.1 0.1 \
        --seed 42
"""

import argparse
import random
import shutil
import sys
from pathlib import Path


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
CLASS_NAMES = ["defective", "non_defective"]


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess & Split Dataset")
    parser.add_argument("--input",  type=str, default="data/raw")
    parser.add_argument("--output", type=str, default="data/processed")
    parser.add_argument("--split",  type=float, nargs=3, default=[0.8, 0.1, 0.1],
                        help="train val test ratios (must sum to 1.0)")
    parser.add_argument("--seed",   type=int, default=42)
    return parser.parse_args()


def split_files(files, ratios, seed):
    random.seed(seed)
    random.shuffle(files)
    n = len(files)
    n_train = int(n * ratios[0])
    n_val   = int(n * ratios[1])
    return {
        "train": files[:n_train],
        "val":   files[n_train : n_train + n_val],
        "test":  files[n_train + n_val :],
    }


def main():
    args = parse_args()
    assert abs(sum(args.split) - 1.0) < 1e-6, "Split ratios must sum to 1.0"

    input_dir  = Path(args.input)
    output_dir = Path(args.output)

    print(f"\nSplitting data: {input_dir} → {output_dir}")
    print(f"Ratios: train={args.split[0]}, val={args.split[1]}, test={args.split[2]}")

    total_copied = 0
    for class_name in CLASS_NAMES:
        src_class_dir = input_dir / class_name
        if not src_class_dir.exists():
            print(f"  [WARN] Class folder not found: {src_class_dir}")
            continue

        files = [
            p for p in src_class_dir.iterdir()
            if p.suffix.lower() in VALID_EXTENSIONS
        ]

        if not files:
            print(f"  [WARN] No images in {src_class_dir}")
            continue

        splits = split_files(files, args.split, args.seed)

        for split_name, split_files_ in splits.items():
            dest = output_dir / split_name / class_name
            dest.mkdir(parents=True, exist_ok=True)
            for fp in split_files_:
                shutil.copy2(fp, dest / fp.name)
            print(f"  {class_name}/{split_name:5s}: {len(split_files_):4d} images → {dest}")
            total_copied += len(split_files_)

    print(f"\n✅ Done. Copied {total_copied} images to {output_dir}")


if __name__ == "__main__":
    main()
