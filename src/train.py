# ----------------------------------------------------------
# Intelligent Image Classification System - Training Script
# Author: Amitoj Singh (CCID: amitoj3)
# ----------------------------------------------------------

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import get_data_loaders
from .model import build_model


def accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    preds = outputs.argmax(dim=1)
    return (preds == targets).float().mean().item()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    pbar = tqdm(loader, desc="Train", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_acc += accuracy(outputs, labels)
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    n = len(loader)
    return running_loss / n, running_acc / n


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            running_acc += accuracy(outputs, labels)
    n = len(loader)
    return running_loss / n, running_acc / n


def save_checkpoint(model: nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def make_optimizer_for_head(model: nn.Module, lr: float) -> torch.optim.Optimizer:
    """Create optimizer for training only the classification head."""
    return Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)


def make_optimizer_head_and_layer4(model: nn.Module, head_lr: float, layer4_lr: float) -> torch.optim.Optimizer:
    """Create optimizer with param groups for head and unfrozen layer4."""
    params = []
    if hasattr(model, "fc"):
        params.append({"params": model.fc.parameters(), "lr": head_lr})
    if hasattr(model, "layer4"):
        params.append({"params": model.layer4.parameters(), "lr": layer4_lr})
    return Adam(params)


def main():
    parser = argparse.ArgumentParser(description="Train ResNet-18 on an image dataset")
    parser.add_argument("--train_dir", type=str, default="data/train", help="Training images root")
    parser.add_argument("--val_dir", type=str, default="data/val", help="Validation images root")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default="models")
    parser.add_argument("--checkpoint_name", type=str, default="resnet18_best.pth")
    parser.add_argument("--unfreeze_after", type=int, default=0, help="Epoch after which to unfreeze layer4 (0=never)")
    parser.add_argument("--unfrozen_lr", type=float, default=1e-4, help="LR for unfrozen layer4 if enabled")

    args = parser.parse_args()

    device = torch.device(args.device)

    # Data
    train_loader, val_loader, num_classes = get_data_loaders(
        args.train_dir, args.val_dir, batch_size=args.batch_size, img_size=args.img_size
    )

    # Persist class mapping for inference later
    class_to_idx = train_loader.dataset.class_to_idx  # type: ignore[attr-defined]
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    classes_path = Path(args.output_dir) / "classes.json"
    classes_path.parent.mkdir(parents=True, exist_ok=True)
    with open(classes_path, "w", encoding="utf-8") as f:
        json.dump(idx_to_class, f, indent=2)

    # Model
    model = build_model(num_classes)
    model.to(device)

    # Train only the head
    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer_for_head(model, lr=args.lr)

    best_val_acc = 0.0
    best_path = Path(args.output_dir) / args.checkpoint_name

    for epoch in range(1, args.epochs + 1):
        # Optionally unfreeze layer4 after warmup
        if args.unfreeze_after and epoch == args.unfreeze_after:
            if hasattr(model, "layer4"):
                for p in model.layer4.parameters():
                    p.requires_grad = True
                optimizer = make_optimizer_head_and_layer4(model, head_lr=args.lr, layer4_lr=args.unfrozen_lr)
                print(
                    f"[Fine-tune] Unfroze layer4 at epoch {epoch}. Head LR={args.lr}, layer4 LR={args.unfrozen_lr}"
                )
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, best_path)
            print(f"Saved best model to {best_path} (val_acc={best_val_acc:.4f})")

    print("Training complete.")


if __name__ == "__main__":
    main()
