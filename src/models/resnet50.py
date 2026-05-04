"""
src/models/resnet50.py
ResNet-50 Transfer Learning model for binary defect classification.

Architecture:
  Input (224x224x3)
    → ResNet-50 Backbone (ImageNet pre-trained)
    → Global Average Pooling (built-in)
    → Custom Classifier Head
      Linear(2048→512) → BN → ReLU → Dropout(0.4)
      Linear(512→128)  → BN → ReLU → Dropout(0.3)
      Linear(128→2)
    → Output: [Non-Defective, Defective]
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional


class DefectClassifier(nn.Module):
    """
    ResNet-50 fine-tuned for binary product defect classification.

    Args:
        num_classes:    Number of output classes (default: 2)
        pretrained:     Use ImageNet pre-trained weights (default: True)
        freeze_layers:  Number of ResNet layer groups to freeze [0–4]
        dropout_rate:   Dropout probability in classifier head
    """

    LAYER_NAMES = ["layer1", "layer2", "layer3", "layer4"]

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        freeze_layers: int = 3,
        dropout_rate: float = 0.4,
    ):
        super().__init__()
        self.num_classes = num_classes

        # ── Backbone ──────────────────────────────────────────────────
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet50(weights=weights)

        # Freeze early layers (stem + layer1..N)
        self._freeze_backbone(backbone, freeze_layers)

        # Keep everything except the final FC layer
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        # Output: (B, 2048, 1, 1)

        # ── Classifier Head ───────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),

            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate * 0.75),

            nn.Linear(128, num_classes),
        )

        self._init_classifier_weights()

    # ------------------------------------------------------------------
    def _freeze_backbone(self, backbone: nn.Module, freeze_layers: int):
        """Freeze stem + first `freeze_layers` ResNet layer groups."""
        # Always freeze the initial stem (conv1, bn1, maxpool)
        for name, param in backbone.named_parameters():
            if any(name.startswith(stem) for stem in ["conv1", "bn1"]):
                param.requires_grad = False

        # Freeze resnet layer groups
        for layer_name in self.LAYER_NAMES[:freeze_layers]:
            for param in getattr(backbone, layer_name).parameters():
                param.requires_grad = False

        n_frozen = sum(not p.requires_grad for p in backbone.parameters())
        n_total  = sum(1 for _ in backbone.parameters())
        print(f"[DefectClassifier] Frozen {n_frozen}/{n_total} backbone params "
              f"({freeze_layers} layer groups)")

    def _init_classifier_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)       # (B, 2048, 1, 1)
        logits   = self.classifier(features)  # (B, num_classes)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns softmax probabilities."""
        return torch.softmax(self.forward(x), dim=1)

    # ------------------------------------------------------------------
    def unfreeze_all(self):
        """Unfreeze all parameters (use for fine-tuning stage 2)."""
        for param in self.parameters():
            param.requires_grad = True
        print("[DefectClassifier] All layers unfrozen.")

    def count_parameters(self) -> dict:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}


# ------------------------------------------------------------------
def build_model(
    num_classes: int = 2,
    pretrained: bool = True,
    freeze_layers: int = 3,
    dropout_rate: float = 0.4,
    device: Optional[str] = None,
) -> DefectClassifier:
    """Factory function — builds and moves model to device."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = DefectClassifier(num_classes, pretrained, freeze_layers, dropout_rate)
    model = model.to(device)

    params = model.count_parameters()
    print(
        f"[build_model] Device: {device} | "
        f"Total params: {params['total']:,} | "
        f"Trainable: {params['trainable']:,}"
    )
    return model


def load_model(
    checkpoint_path: str,
    device: Optional[str] = None,
    num_classes: int = 2,
) -> DefectClassifier:
    """Load model from a saved checkpoint."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = DefectClassifier(num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"[load_model] Loaded checkpoint from '{checkpoint_path}' "
          f"(epoch {checkpoint.get('epoch', '?')}, "
          f"val_acc={checkpoint.get('val_acc', 0):.4f})")
    return model
