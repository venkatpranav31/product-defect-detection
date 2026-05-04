"""
src/inference/pipeline.py
Real-time inference pipeline targeting 20 images/second.

Features:
  - Single image inference
  - Batch folder inference
  - Confidence thresholding
  - FPS benchmarking
  - Result logging to JSON/CSV
"""

import time
import json
import csv
from pathlib import Path
from typing import Union, List, Optional

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from src.data.augmentation import get_inference_transforms


CLASS_NAMES = {0: "Non-Defective", 1: "Defective"}
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


class InferencePipeline:
    """
    High-speed inference pipeline for manufacturing QC.

    Targets ≥20 images/second on GPU.
    On CPU it gracefully degrades with smaller batches.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        image_size: int = 224,
        confidence_threshold: float = 0.70,
        batch_size: int = 16,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = model.to(self.device)
        self.model.eval()

        self.transform = get_inference_transforms(image_size)
        self.confidence_threshold = confidence_threshold
        self.batch_size = batch_size

        print(
            f"[InferencePipeline] device={self.device} | "
            f"batch_size={batch_size} | "
            f"conf_threshold={confidence_threshold}"
        )

    # ------------------------------------------------------------------
    def predict_single(self, image_path: Union[str, Path]) -> dict:
        """
        Classify a single image.

        Returns:
            {
                "path":         str,
                "label":        "Defective" | "Non-Defective",
                "label_id":     0 | 1,
                "confidence":   float,
                "is_confident": bool,
                "latency_ms":   float,
            }
        """
        image_path = Path(image_path)
        img = Image.open(image_path).convert("RGB")
        tensor = self.transform(img).unsqueeze(0).to(self.device)

        t0 = time.perf_counter()
        with torch.no_grad():
            logits = self.model(tensor)
            probs  = torch.softmax(logits, dim=1)[0]
        latency_ms = (time.perf_counter() - t0) * 1000

        label_id   = int(probs.argmax().item())
        confidence = float(probs[label_id].item())

        return {
            "path":         str(image_path),
            "label":        CLASS_NAMES[label_id],
            "label_id":     label_id,
            "confidence":   round(confidence, 4),
            "is_confident": confidence >= self.confidence_threshold,
            "latency_ms":   round(latency_ms, 2),
        }

    # ------------------------------------------------------------------
    def predict_batch(
        self,
        image_paths: List[Union[str, Path]],
        save_results: Optional[str] = None,
    ) -> List[dict]:
        """
        Run batch inference on a list of image paths.
        Processes images in mini-batches for throughput efficiency.
        """
        results = []
        total_time = 0.0

        for i in tqdm(
            range(0, len(image_paths), self.batch_size),
            desc="Inferencing batches",
        ):
            batch_paths = image_paths[i : i + self.batch_size]
            tensors = []
            valid_paths = []

            for p in batch_paths:
                try:
                    img = Image.open(p).convert("RGB")
                    tensors.append(self.transform(img))
                    valid_paths.append(str(p))
                except Exception as e:
                    print(f"  [WARN] Skipping {p}: {e}")

            if not tensors:
                continue

            batch = torch.stack(tensors).to(self.device)

            t0 = time.perf_counter()
            with torch.no_grad():
                logits = self.model(batch)
                probs  = torch.softmax(logits, dim=1)
            elapsed = time.perf_counter() - t0
            total_time += elapsed

            for j, path in enumerate(valid_paths):
                label_id   = int(probs[j].argmax().item())
                confidence = float(probs[j][label_id].item())
                results.append({
                    "path":         path,
                    "label":        CLASS_NAMES[label_id],
                    "label_id":     label_id,
                    "confidence":   round(confidence, 4),
                    "is_confident": confidence >= self.confidence_threshold,
                })

        # FPS Report
        if total_time > 0:
            fps = len(results) / total_time
            print(f"\n[Pipeline] {len(results)} images in {total_time:.2f}s → {fps:.1f} FPS")

        # Save results
        if save_results and results:
            self._save_results(results, save_results)

        return results

    # ------------------------------------------------------------------
    def run_on_folder(
        self,
        folder: Union[str, Path],
        save_results: Optional[str] = None,
    ) -> List[dict]:
        """Discover all images in a folder and run batch inference."""
        folder = Path(folder)
        paths  = [
            p for p in sorted(folder.rglob("*"))
            if p.suffix.lower() in VALID_EXTENSIONS
        ]

        if not paths:
            raise FileNotFoundError(f"No images found in {folder}")

        print(f"[Pipeline] Found {len(paths)} images in {folder}")
        return self.predict_batch(paths, save_results=save_results)

    # ------------------------------------------------------------------
    def benchmark(self, num_images: int = 100, image_size: int = 224) -> dict:
        """Benchmark throughput using random synthetic images."""
        dummy = torch.randn(self.batch_size, 3, image_size, image_size).to(self.device)

        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = self.model(dummy)

        # Benchmark
        if self.device == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        n_batches = max(1, num_images // self.batch_size)

        with torch.no_grad():
            for _ in range(n_batches):
                _ = self.model(dummy)
                if self.device == "cuda":
                    torch.cuda.synchronize()

        total   = time.perf_counter() - t0
        n_imgs  = n_batches * self.batch_size
        fps     = n_imgs / total
        lat_ms  = (total / n_imgs) * 1000

        stats = {
            "fps":              round(fps, 2),
            "latency_ms":       round(lat_ms, 2),
            "device":           self.device,
            "batch_size":       self.batch_size,
            "meets_target_20fps": fps >= 20.0,
        }
        print(f"\n[Benchmark] {fps:.1f} FPS  |  {lat_ms:.2f} ms/image  |  target=20 FPS → {'✅ PASS' if fps >= 20 else '❌ FAIL'}")
        return stats

    # ------------------------------------------------------------------
    @staticmethod
    def _save_results(results: list, path: str):
        path = Path(path)
        if path.suffix == ".json":
            with open(path, "w") as f:
                json.dump(results, f, indent=2)
        else:
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
        print(f"[Pipeline] Results saved → {path}")
