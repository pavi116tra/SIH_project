import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from model import X3DFightClassifier
from dataset import X3DVideoDataset

def train_model(splits_file="splits.json", epochs=5, batch_size=4, lr=1e-4, checkpoint_path="fight_detector_x3d_s.pt"):
    """
    Main training/fine-tuning loop for X3D-S Fight Classifier.
    Saves best model weights to fight_detector_x3d_s.pt.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for X3D-S fine-tuning: {device}")

    if not os.path.exists(splits_file):
        print(f"Error: Splits file '{splits_file}' not found! Run data_cleaner.py first.")
        return

    with open(splits_file, "r") as f:
        splits = json.load(f)

    train_samples = splits["train"]
    val_samples = splits["val"]

    print(f"Loaded {len(train_samples)} training video clips, {len(val_samples)} validation video clips.")

    train_dataset = X3DVideoDataset(train_samples, num_frames=16, is_train=True)
    val_dataset = X3DVideoDataset(val_samples, num_frames=16, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = X3DFightClassifier(pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_f1 = 0.0

    print("\nStarting X3D-S Training Loop...")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        train_preds, train_targets = [], []

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device) # Inputs shape: (B, 3, 16, 224, 224)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            train_preds.extend(preds)
            train_targets.extend(targets.cpu().numpy())

        scheduler.step()
        epoch_train_loss = running_loss / len(train_dataset)
        epoch_train_f1 = f1_score(train_targets, train_preds, zero_division=0)
        epoch_train_acc = accuracy_score(train_targets, train_preds)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item() * inputs.size(0)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(targets.cpu().numpy())

        epoch_val_loss = val_loss / len(val_dataset)
        epoch_val_f1 = f1_score(val_targets, val_preds, zero_division=0)
        epoch_val_acc = accuracy_score(val_targets, val_preds)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] "
              f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f} | Train F1: {epoch_train_f1:.4f} || "
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f} | Val F1: {epoch_val_f1:.4f}")

        # Save Best Checkpoint
        if epoch_val_f1 >= best_val_f1:
            best_val_f1 = epoch_val_f1
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  --> Saved new best X3D-S checkpoint to '{checkpoint_path}' (Val F1: {best_val_f1:.4f})")

    print(f"\nX3D-S Training completed! Best Validation F1 Score: {best_val_f1:.4f}")
    return checkpoint_path

if __name__ == "__main__":
    train_model(epochs=3)
