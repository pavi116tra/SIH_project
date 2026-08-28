import os
import json
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from model import X3DFightClassifier
from dataset import X3DVideoDataset

def evaluate_test_set(checkpoint_path="fight_detector_x3d_s.pt", splits_file="splits.json"):
    """
    Evaluates trained X3DFightClassifier on held-out test split.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating X3D-S model checkpoint '{checkpoint_path}' on device: {device}")

    if not os.path.exists(splits_file):
        print(f"Error: Splits file '{splits_file}' not found!")
        return

    with open(splits_file, "r") as f:
        splits = json.load(f)

    test_samples = splits["test"]
    print(f"Loaded {len(test_samples)} held-out test video samples.")

    test_dataset = X3DVideoDataset(test_samples, num_frames=16, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    model = X3DFightClassifier(pretrained=False).to(device)
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded checkpoint weights from '{checkpoint_path}'")
    else:
        print(f"Warning: Checkpoint '{checkpoint_path}' not found! Evaluating initialized model.")

    model.eval()

    all_probabilities = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy() # Probability of fight class (index 1)
            all_probabilities.extend(probs)
            all_targets.extend(targets.numpy())

    all_probabilities = np.array(all_probabilities)
    all_targets = np.array(all_targets)

    # Threshold Tuning (0.10 to 0.95)
    best_thresh = 0.60
    best_f1 = 0.0

    print("\n--- Confidence Threshold Tuning (Target Threshold = 0.60) ---")
    for thresh in np.arange(0.1, 0.95, 0.05):
        preds = (all_probabilities >= thresh).astype(int)
        f1 = f1_score(all_targets, preds, zero_division=0)
        print(f"Threshold: {thresh:.2f} | Test F1 Score: {f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    # Evaluate specifically at target threshold 0.60
    preds_06 = (all_probabilities >= 0.60).astype(int)
    f1_06 = f1_score(all_targets, preds_06, zero_division=0)
    print(f"\nTarget Threshold 0.60 F1 Score: {f1_06:.4f} (Best Threshold: {best_thresh:.2f} with F1: {best_f1:.4f})")

    report = classification_report(all_targets, preds_06, labels=[0, 1], target_names=["no_fight", "fight"], digits=4, zero_division=0)
    cm = confusion_matrix(all_targets, preds_06, labels=[0, 1])

    print("\n--- X3D-S Held-Out Test Set Classification Report (Threshold 0.60) ---")
    print(report)
    print("Confusion Matrix:")
    print(f"  [TN: {cm[0,0]}  FP: {cm[0,1]}]")
    print(f"  [FN: {cm[1,0]}  TP: {cm[1,1]}]")

    return 0.60, f1_06

if __name__ == "__main__":
    evaluate_test_set()
