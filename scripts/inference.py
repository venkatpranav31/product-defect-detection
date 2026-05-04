"""
scripts/inference.py
Entry point for single image or batch folder inference.

Usage:
    # Single image
    python scripts/inference.py --image path/to/image.jpg --model outputs/models/best_model.pth

    # Entire folder
    python scripts/inference.py --folder path/to/images/ --model outputs/models/best_model.pth

    # Benchmark throughput
    python scripts/inference.py --benchmark --model outputs/models/best_model.pth
"""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.resnet50 import load_model
from src.inference.pipeline import InferencePipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Run Defect Detection Inference")
    parser.add_argument("--model",      type=str, required=True)
    parser.add_argument("--config",     type=str, default="config/config.yaml")
    parser.add_argument("--image",      type=str, default=None, help="Single image path")
    parser.add_argument("--folder",     type=str, default=None, help="Folder of images")
    parser.add_argument("--output",     type=str, default=None, help="Save results to JSON/CSV")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--threshold",  type=float, default=None)
    parser.add_argument("--benchmark",  action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    infer_cfg = cfg.get("inference", {})
    batch_size = args.batch_size or infer_cfg.get("batch_size", 16)
    threshold  = args.threshold  or infer_cfg.get("confidence_threshold", 0.70)
    img_size   = cfg["data"]["image_size"]

    model    = load_model(args.model, num_classes=cfg["model"]["num_classes"])
    pipeline = InferencePipeline(
        model=model,
        image_size=img_size,
        confidence_threshold=threshold,
        batch_size=batch_size,
    )

    if args.benchmark:
        pipeline.benchmark(num_images=200)

    elif args.image:
        result = pipeline.predict_single(args.image)
        print(f"\n  Image  : {result['path']}")
        print(f"  Label  : {result['label']}")
        print(f"  Conf   : {result['confidence']:.4f}  ({'✅ confident' if result['is_confident'] else '⚠️  low confidence'})")
        print(f"  Latency: {result['latency_ms']:.2f} ms")

    elif args.folder:
        results = pipeline.run_on_folder(args.folder, save_results=args.output)

        defective = sum(1 for r in results if r["label_id"] == 1)
        total     = len(results)
        print(f"\n  Inspected : {total} images")
        print(f"  Defective : {defective} ({defective/total*100:.1f}%)")
        print(f"  Non-Defective: {total - defective} ({(total-defective)/total*100:.1f}%)")

    else:
        print("Please provide --image, --folder, or --benchmark")


if __name__ == "__main__":
    main()
