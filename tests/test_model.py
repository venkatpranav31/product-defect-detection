"""
tests/test_model.py
Unit tests for the DefectClassifier model.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch
from src.models.resnet50 import DefectClassifier, build_model


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def model():
    return DefectClassifier(
        num_classes=2,
        pretrained=False,   # avoid downloading weights during CI
        freeze_layers=0,
        dropout_rate=0.4,
    ).to(DEVICE)


class TestDefectClassifier:

    def test_output_shape(self, model):
        """Model should output (B, 2) for any batch size."""
        x = torch.randn(4, 3, 224, 224).to(DEVICE)
        out = model(x)
        assert out.shape == (4, 2), f"Expected (4,2), got {out.shape}"

    def test_predict_proba_sums_to_one(self, model):
        """Softmax probabilities must sum to 1."""
        model.eval()
        x = torch.randn(8, 3, 224, 224).to(DEVICE)
        probs = model.predict_proba(x)
        sums = probs.sum(dim=1)
        assert torch.allclose(sums, torch.ones(8).to(DEVICE), atol=1e-5)

    def test_freeze_layers(self):
        """Frozen parameters should have requires_grad=False."""
        m = DefectClassifier(pretrained=False, freeze_layers=3)
        n_frozen = sum(1 for p in m.parameters() if not p.requires_grad)
        assert n_frozen > 0, "Expected some frozen params"

    def test_unfreeze_all(self):
        """After unfreeze_all(), all params should be trainable."""
        m = DefectClassifier(pretrained=False, freeze_layers=4)
        m.unfreeze_all()
        n_frozen = sum(1 for p in m.parameters() if not p.requires_grad)
        assert n_frozen == 0

    def test_count_parameters(self, model):
        counts = model.count_parameters()
        assert counts["total"] > 0
        assert counts["trainable"] + counts["frozen"] == counts["total"]

    def test_single_image_inference(self, model):
        """Inference on a single image should work."""
        model.eval()
        x = torch.randn(1, 3, 224, 224).to(DEVICE)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 2)

    def test_different_batch_sizes(self, model):
        """Model must handle variable batch sizes."""
        model.eval()
        for bs in [1, 4, 16, 32]:
            x = torch.randn(bs, 3, 224, 224).to(DEVICE)
            with torch.no_grad():
                out = model(x)
            assert out.shape == (bs, 2)
