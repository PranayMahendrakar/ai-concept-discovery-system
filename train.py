"""
train.py - Self-Supervised Training Script
==========================================
Train the SimCLR encoder on unlabeled images to learn visual concepts.

Usage:
    # Train on synthetic shapes (quick demo)
    python train.py --dataset synthetic --n_samples 2000 --epochs 50

    # Train on a folder of images
    python train.py --dataset folder --data_path ./data/images --epochs 100

    # Train on STL-10 unlabeled set
    python train.py --dataset stl10 --data_path ./data --epochs 200 --batch_size 64
"""

import argparse
import os
import sys
import json
import torch

from concept_discovery.dataset import get_dataset, get_dataloader
from concept_discovery.trainer import SimCLRTrainer
from concept_discovery.visualizer import ConceptVisualizer


def parse_args():
    parser = argparse.ArgumentParser(description="Train SimCLR for self-supervised concept discovery")
    parser.add_argument("--dataset", type=str, default="synthetic",
                        choices=["synthetic", "folder", "stl10", "cifar10"],
                        help="Dataset type")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to image folder (for 'folder', 'stl10', 'cifar10' datasets)")
    parser.add_argument("--n_samples", type=int, default=2000,
                        help="Number of synthetic samples (for 'synthetic' dataset)")
    parser.add_argument("--image_size", type=int, default=64,
                        help="Input image size")
    parser.add_argument("--backbone", type=str, default="resnet18",
                        choices=["resnet18", "resnet50"],
                        help="Encoder backbone architecture")
    parser.add_argument("--projection_dim", type=int, default=128,
                        help="Projection head output dimension")
    parser.add_argument("--temperature", type=float, default=0.5,
                        help="NT-Xent loss temperature")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Training batch size")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay")
    parser.add_argument("--warmup", type=int, default=5,
                        help="Warmup epochs")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Output directory for checkpoints and plots")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu), auto-detected if not specified")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader workers")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  AI Concept Discovery System - SimCLR Training")
    print("=" * 60)
    print(f"  Dataset:     {args.dataset}")
    print(f"  Backbone:    {args.backbone}")
    print(f"  Epochs:      {args.epochs}")
    print(f"  Batch size:  {args.batch_size}")
    print(f"  Image size:  {args.image_size}x{args.image_size}")
    print(f"  Output dir:  {args.output_dir}")
    print()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Dataset
    print("[1/4] Loading dataset...")
    train_dataset = get_dataset(
        dataset_type=args.dataset,
        root=args.data_path,
        image_size=args.image_size,
        n_samples=args.n_samples,
        mode="train",
    )
    train_loader = get_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Batches/epoch: {len(train_loader)}")

    # 2. Trainer
    print("\n[2/4] Initializing SimCLR Trainer...")
    trainer = SimCLRTrainer(
        backbone=args.backbone,
        projection_dim=args.projection_dim,
        temperature=args.temperature,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        warmup_epochs=args.warmup,
        image_size=args.image_size,
        checkpoint_dir=os.path.join(args.output_dir, "checkpoints"),
        log_dir=os.path.join(args.output_dir, "logs"),
        device=args.device,
    )
    print(trainer.config_summary())

    # 3. Train
    print("\n[3/4] Training...")
    history = trainer.train(train_loader)

    # 4. Visualize loss curve
    print("\n[4/4] Generating visualizations...")
    viz = ConceptVisualizer(output_dir=os.path.join(args.output_dir, "plots"))
    loss_path = viz.plot_loss_curve(
        losses=history["train_losses"],
        title=f"SimCLR NT-Xent Loss ({args.backbone}, temp={args.temperature})",
    )

    # Save final config
    config = vars(args)
    config["best_loss"] = history["best_loss"]
    config_path = os.path.join(args.output_dir, "training_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print()
    print("=" * 60)
    print("  Training Complete!")
    print(f"  Best Loss:       {history['best_loss']:.4f}")
    print(f"  Checkpoint:      {trainer.checkpoint_dir}/best_model.pt")
    print(f"  Loss curve:      {loss_path}")
    print(f"  Config:          {config_path}")
    print()
    print("  Next step: Run discover_concepts.py to find visual concepts!")
    print("  Example:")
    print(f"    python discover_concepts.py --checkpoint {trainer.checkpoint_dir}/best_model.pt")
    print("=" * 60)


if __name__ == "__main__":
    main()
