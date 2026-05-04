"""
src/evaluation/evaluator.py
Model evaluation:
  - Accuracy, Precision, Recall, F1 (per-class & macro)
  - Confusion matrix heatmap
  - ROC curve & AUC
  - Full classification report saved to disk
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve, auc,
    f1_score,
)
from tqdm import tqdm


CLASS_NAMES = ["Non-Defective", "Defective"]


class Evaluator:
    """
    Evaluates a trained model on a DataLoader.
    Generates all metrics + publication-ready plots.
    """

    def __init__(
        self,
        model: nn.Module,
        loader,
        output_dir: str = "outputs/reports",
        device: Optional[str] = None,
    ):
        self.model  = model
        self.loader = loader
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model.to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------
    def _collect_predictions(self):
        all_labels, all_preds, all_probs = [], [], []

        with torch.no_grad():
            for images, labels in tqdm(self.loader, desc="Evaluating"):
                images = images.to(self.device)
                logits = self.model(images)
                probs  = torch.softmax(logits, dim=1)
                preds  = probs.argmax(dim=1)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        return (
            np.array(all_labels),
            np.array(all_preds),
            np.array(all_probs),
        )

    # ------------------------------------------------------------------
    def evaluate(self) -> dict:
        labels, preds, probs = self._collect_predictions()

        accuracy = accuracy_score(labels, preds)
        f1_macro = f1_score(labels, preds, average="macro")
        report   = classification_report(labels, preds, target_names=CLASS_NAMES, output_dict=True)

        metrics = {
            "accuracy":    round(float(accuracy), 4),
            "f1_macro":    round(float(f1_macro), 4),
            "per_class":   report,
        }

        print(f"\n{'='*50}")
        print(f"  Accuracy  : {accuracy:.4f}")
        print(f"  F1 Macro  : {f1_macro:.4f}")
        print(f"\n{classification_report(labels, preds, target_names=CLASS_NAMES)}")

        # Save JSON report
        report_path = self.output_dir / "metrics.json"
        with open(report_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"  Metrics saved → {report_path}")

        # Plots
        self._plot_confusion_matrix(labels, preds)
        self._plot_roc_curve(labels, probs)

        return metrics

    # ------------------------------------------------------------------
    def _plot_confusion_matrix(self, labels: np.ndarray, preds: np.ndarray):
        cm = confusion_matrix(labels, preds)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, data, fmt, title in zip(
            axes,
            [cm, cm_norm],
            ["d", ".2%"],
            ["Confusion Matrix (counts)", "Confusion Matrix (normalized)"]
        ):
            sns.heatmap(
                data, annot=True, fmt=fmt, cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=ax, linewidths=0.5, cbar_kws={"shrink": 0.8}
            )
            ax.set_title(title, fontsize=13, fontweight="bold")
            ax.set_ylabel("True Label")
            ax.set_xlabel("Predicted Label")

        plt.tight_layout()
        path = self.output_dir / "confusion_matrix.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Confusion matrix saved → {path}")

    def _plot_roc_curve(self, labels: np.ndarray, probs: np.ndarray):
        defect_probs = probs[:, 1]   # probability of "defective"
        fpr, tpr, _ = roc_curve(labels, defect_probs)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(fpr, tpr, color="#2563eb", lw=2,
                label=f"ROC curve (AUC = {roc_auc:.4f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
        ax.fill_between(fpr, tpr, alpha=0.1, color="#2563eb")

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title("ROC Curve — Defect Detection", fontsize=13, fontweight="bold")
        ax.legend(loc="lower right", fontsize=11)
        ax.grid(alpha=0.3)

        path = self.output_dir / "roc_curve.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  ROC curve saved → {path}  (AUC={roc_auc:.4f})")
