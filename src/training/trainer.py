"""
src/training/trainer.py
Full training loop with:
  - Mixed precision (AMP)
  - Early stopping
  - LR scheduling (cosine annealing)
  - TensorBoard logging
  - Best-checkpoint saving
"""

import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class EarlyStopping:
    """Stops training when val loss stops improving."""

    def __init__(self, patience: int = 5, delta: float = 1e-4):
        self.patience = patience
        self.delta = delta
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class Trainer:
    """
    Manages the full training & validation loop.

    Usage:
        trainer = Trainer(model, loaders, config)
        trainer.train()
    """

    def __init__(
        self,
        model: nn.Module,
        loaders: dict,
        config: dict,
        output_dir: str = "outputs/models",
        log_dir: str = "outputs/logs",
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.loaders = loaders
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # ── Optimizer ────────────────────────────────────────────────
        lr = config.get("learning_rate", 1e-3)
        wd = config.get("weight_decay", 1e-4)
        opt_name = config.get("optimizer", "adamw").lower()

        if opt_name == "adamw":
            self.optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=lr, weight_decay=wd
            )
        elif opt_name == "adam":
            self.optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=lr, weight_decay=wd
            )
        else:
            self.optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=lr, momentum=0.9, weight_decay=wd
            )

        # ── Scheduler ────────────────────────────────────────────────
        epochs = config.get("epochs", 30)
        sched_name = config.get("scheduler", "cosine").lower()
        if sched_name == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=epochs, eta_min=1e-6
            )
        elif sched_name == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=10, gamma=0.1
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", patience=3, factor=0.5
            )

        # ── Loss ─────────────────────────────────────────────────────
        # Use class weights if provided
        class_weights = config.get("class_weights", None)
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)

        # ── AMP ──────────────────────────────────────────────────────
        self.use_amp = config.get("mixed_precision", True) and self.device == "cuda"
        self.scaler  = GradScaler() if self.use_amp else None

        # ── Early Stopping ───────────────────────────────────────────
        patience = config.get("patience", 5)
        self.early_stopping = EarlyStopping(patience=patience)

        # ── TensorBoard ──────────────────────────────────────────────
        self.writer = SummaryWriter(log_dir=log_dir)

        # ── State ────────────────────────────────────────────────────
        self.best_val_acc = 0.0
        self.history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # ------------------------------------------------------------------
    def _run_epoch(self, split: str) -> tuple:
        is_train = split == "train"
        self.model.train() if is_train else self.model.eval()
        loader = self.loaders[split]

        total_loss, correct, total = 0.0, 0, 0

        ctx = torch.enable_grad() if is_train else torch.no_grad()
        with ctx:
            pbar = tqdm(loader, desc=f"  {split.capitalize():5s}", leave=False)
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)

                if is_train:
                    self.optimizer.zero_grad(set_to_none=True)

                if self.use_amp and is_train:
                    with autocast():
                        logits = self.model(images)
                        loss   = self.criterion(logits, labels)
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    logits = self.model(images)
                    loss   = self.criterion(logits, labels)
                    if is_train:
                        loss.backward()
                        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                        self.optimizer.step()

                preds       = logits.argmax(dim=1)
                correct    += (preds == labels).sum().item()
                total      += labels.size(0)
                total_loss += loss.item() * labels.size(0)

                pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.4f}")

        return total_loss / total, correct / total

    # ------------------------------------------------------------------
    def _save_checkpoint(self, epoch: int, val_acc: float):
        path = self.output_dir / "best_model.pth"
        torch.save({
            "epoch":            epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state":  self.optimizer.state_dict(),
            "val_acc":          val_acc,
        }, path)
        print(f"  ✅  Saved best model → {path}  (val_acc={val_acc:.4f})")

    # ------------------------------------------------------------------
    def train(self):
        epochs = self.config.get("epochs", 30)
        print(f"\n{'='*60}")
        print(f"  Training on {self.device.upper()} | {epochs} epochs | AMP={self.use_amp}")
        print(f"{'='*60}\n")

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            print(f"Epoch [{epoch:02d}/{epochs}]")

            train_loss, train_acc = self._run_epoch("train")
            val_loss,   val_acc   = self._run_epoch("val")

            # Scheduler step
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

            lr = self.optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0

            # Logging
            self.writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch)
            self.writer.add_scalars("Acc",  {"train": train_acc,  "val": val_acc},  epoch)
            self.writer.add_scalar("LR", lr, epoch)

            for key, val in [("train_loss", train_loss), ("train_acc", train_acc),
                              ("val_loss",   val_loss),   ("val_acc",   val_acc)]:
                self.history[key].append(val)

            print(
                f"  Train  loss={train_loss:.4f}  acc={train_acc:.4f}\n"
                f"  Val    loss={val_loss:.4f}  acc={val_acc:.4f}  "
                f"lr={lr:.2e}  [{elapsed:.1f}s]"
            )

            # Checkpoint
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self._save_checkpoint(epoch, val_acc)

            # Early stopping
            if self.early_stopping(val_loss):
                print(f"\n  ⏹  Early stopping triggered at epoch {epoch}.")
                break

            print()

        self.writer.close()
        print(f"\n✅ Training complete. Best val accuracy: {self.best_val_acc:.4f}")
        return self.history
