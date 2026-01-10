import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np
from tqdm import tqdm
from typing import Tuple
from losses import AdCorreLoss
import warnings
import config
from dataset import get_data_loaders
from models import EfficientNetEmotionClassifier
from utils import save_checkpoint, EarlyStopping


warnings.filterwarnings("ignore")


def train_epoch(
    model: nn.Module,
    train_loader,
    criterion,
    optimizer,
    device: torch.device,
    epoch: int,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        features = model.get_features(inputs)

        loss = criterion(outputs, targets, features, epoch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        pbar.set_postfix(
            {"loss": running_loss / (batch_idx + 1), "acc": 100.0 * correct / total}
        )

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc


def validate(
    model: nn.Module, val_loader, criterion, device: torch.device, epoch: int = 0
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc="Validating"):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            features = model.get_features(inputs)
            loss = criterion(outputs, targets, features, epoch)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc, np.array(all_preds), np.array(all_targets)


def train():
    cfg = config.Config

    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    torch.manual_seed(42)
    np.random.seed(42)

    train_loader, val_loader = get_data_loaders()

    print(f"Starting training on {cfg.DEVICE}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Dataset: {cfg.HF_DATASET_NAME}")

    model = EfficientNetEmotionClassifier(
        num_classes=cfg.NUM_CLASSES, pretrained=cfg.PRETRAINED
    ).to(cfg.DEVICE)

    criterion = AdCorreLoss(initial_lambda=0.4, num_classes=cfg.NUM_CLASSES)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY,
        betas=(0.9, 0.999),
    )

    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1, eta_min=1e-6)

    early_stopping = EarlyStopping(patience=10, delta=0.001, verbose=True)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    best_acc = 0.0

    print(f"Starting training on {cfg.DEVICE}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")

    for epoch in range(cfg.NUM_EPOCHS):

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, cfg.DEVICE, epoch
        )

        val_loss, val_acc, _, _ = validate(
            model, val_loader, criterion, cfg.DEVICE, epoch
        )

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"\nEpoch {epoch+1}/{cfg.NUM_EPOCHS}:")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_acc": val_acc,
                    "best_acc": best_acc,
                    "history": history,
                    "config": cfg.to_dict(),
                },
                filename=cfg.MODEL_SAVE_PATH,
            )
            print(f"Best model saved with accuracy: {best_acc:.2f}%")

        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    print(f"\nTraining completed! Best validation accuracy: {best_acc:.2f}%")

    from utils import plot_training_history

    plot_training_history(history)


if __name__ == "__main__":
    train()
