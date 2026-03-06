"""
SimCLR Trainer
==============
Self-supervised training loop for concept discovery.
No labels needed - learns from augmented image pairs.
"""

import os
import time
import math
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from typing import Optional, Dict, List
import json

from .simclr_model import SimCLREncoder, NTXentLoss
from .augmentations import SimCLRAugmentation


class CosineAnnealingWarmup:
    """Learning rate scheduler: linear warmup + cosine annealing."""
    def __init__(self, optimizer, warmup_epochs: int, max_epochs: int, base_lr: float, min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr

    def step(self, epoch: int):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr


class SimCLRTrainer:
    """
    Trains SimCLR encoder using self-supervised contrastive learning.

    The model learns WITHOUT labels by:
        1. Taking an image
        2. Applying two different random augmentations -> (view_1, view_2)
        3. Maximizing agreement between views of the same image (positive pair)
        4. Minimizing agreement with views from different images (negative pairs)

    After training, the encoder is used to extract concept embeddings.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        projection_dim: int = 128,
        temperature: float = 0.5,
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-4,
        batch_size: int = 32,
        num_epochs: int = 100,
        warmup_epochs: int = 10,
        image_size: int = 64,
        checkpoint_dir: str = "outputs/checkpoints",
        log_dir: str = "outputs/logs",
        device: Optional[str] = None,
        mixed_precision: bool = True,
    ):
        self.backbone = backbone
        self.projection_dim = projection_dim
        self.temperature = temperature
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.warmup_epochs = warmup_epochs
        self.image_size = image_size
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.mixed_precision = mixed_precision

        # Device setup
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"  [Trainer] Device: {self.device}")

        # Create output dirs
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        # Model, loss, optimizer
        self.model = SimCLREncoder(backbone=backbone, projection_dim=projection_dim).to(self.device)
        self.criterion = NTXentLoss(temperature=temperature, device=str(self.device))
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.scheduler = CosineAnnealingWarmup(
            self.optimizer, warmup_epochs=warmup_epochs, max_epochs=num_epochs, base_lr=learning_rate
        )
        self.scaler = GradScaler(enabled=mixed_precision and self.device.type == "cuda")

        # History
        self.train_losses: List[float] = []
        self.learning_rates: List[float] = []
        self.best_loss = float("inf")
        self.epoch = 0

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Run one epoch of self-supervised training."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            # Unpack two augmented views
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                x_i, x_j = batch[0].to(self.device), batch[1].to(self.device)
            else:
                # Single image batch - generate views on-the-fly
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                aug = SimCLRAugmentation(self.image_size)
                # This branch is for testing; normally loader provides views
                x_i = x.to(self.device)
                x_j = x.to(self.device)

            self.optimizer.zero_grad()

            if self.mixed_precision and self.device.type == "cuda":
                with autocast():
                    _, z_i = self.model(x_i)
                    _, z_j = self.model(x_j)
                    loss = self.criterion(z_i, z_j)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                _, z_i = self.model(x_i)
                _, z_j = self.model(x_j)
                loss = self.criterion(z_i, z_j)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def train(self, dataloader: DataLoader, val_dataloader: Optional[DataLoader] = None) -> Dict:
        """
        Full training loop.

        Args:
            dataloader:     Training DataLoader (returns augmented pairs)
            val_dataloader: Optional validation DataLoader

        Returns:
            Training history dict with losses and learning rates
        """
        print(f"  Starting SimCLR training for {self.num_epochs} epochs...")
        print(f"  Model: {self.backbone} | Batch: {self.batch_size} | LR: {self.learning_rate}")
        print(f"  Temperature: {self.temperature} | Projection dim: {self.projection_dim}")

        start_time = time.time()

        for epoch in range(self.num_epochs):
            self.epoch = epoch
            lr = self.scheduler.step(epoch)
            self.learning_rates.append(lr)

            epoch_loss = self.train_epoch(dataloader)
            self.train_losses.append(epoch_loss)

            # Logging
            elapsed = time.time() - start_time
            eta = elapsed / (epoch + 1) * (self.num_epochs - epoch - 1)

            if (epoch + 1) % max(1, self.num_epochs // 20) == 0 or epoch == 0:
                print(
                    f"  Epoch [{epoch + 1:3d}/{self.num_epochs}] "
                    f"Loss: {epoch_loss:.4f} | LR: {lr:.2e} | "
                    f"Elapsed: {elapsed/60:.1f}m | ETA: {eta/60:.1f}m"
                )

            # Save best checkpoint
            if epoch_loss < self.best_loss:
                self.best_loss = epoch_loss
                self.save_checkpoint("best_model.pt")

        # Save final checkpoint
        self.save_checkpoint("final_model.pt")

        # Save training history
        history = {
            "train_losses": self.train_losses,
            "learning_rates": self.learning_rates,
            "best_loss": self.best_loss,
            "epochs": self.num_epochs,
            "backbone": self.backbone,
            "temperature": self.temperature,
        }
        hist_path = os.path.join(self.log_dir, "training_history.json")
        with open(hist_path, "w") as f:
            json.dump(history, f, indent=2)

        print(f"  Training complete! Best loss: {self.best_loss:.4f}")
        print(f"  History saved: {hist_path}")
        return history

    @torch.no_grad()
    def extract_embeddings(self, dataloader: DataLoader) -> torch.Tensor:
        """
        Extract feature embeddings from all images in dataloader.
        Used after training to discover concepts via clustering.

        Returns:
            embeddings: (N, feature_dim) tensor
        """
        self.model.eval()
        all_embeddings = []

        for batch in dataloader:
            x = batch[0].to(self.device) if isinstance(batch, (list, tuple)) else batch.to(self.device)
            h, _ = self.model(x)
            all_embeddings.append(h.cpu())

        return torch.cat(all_embeddings, dim=0)

    def save_checkpoint(self, filename: str):
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save({
            "epoch": self.epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": self.best_loss,
            "config": {
                "backbone": self.backbone,
                "projection_dim": self.projection_dim,
                "temperature": self.temperature,
            }
        }, path)

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epoch = checkpoint.get("epoch", 0)
        self.best_loss = checkpoint.get("loss", float("inf"))
        print(f"  Loaded checkpoint from {path} (epoch {self.epoch})")

    def get_model(self) -> SimCLREncoder:
        return self.model

    def config_summary(self) -> str:
        lines = [
            "=== SimCLR Trainer Config ===",
            f"Backbone:       {self.backbone}",
            f"Projection dim: {self.projection_dim}",
            f"Temperature:    {self.temperature}",
            f"Learning rate:  {self.learning_rate}",
            f"Batch size:     {self.batch_size}",
            f"Epochs:         {self.num_epochs}",
            f"Warmup epochs:  {self.warmup_epochs}",
            f"Image size:     {self.image_size}",
            f"Device:         {self.device}",
            f"Mixed precision:{self.mixed_precision}",
        ]
        return "\n".join(lines)
