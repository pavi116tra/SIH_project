"""
Dataset Training Script for Fine-Tuning YOLO Object Detector on Custom Surveillance Data.
Supports low-light, occluded, long-distance, and multi-object environment training.
"""

import argparse
import os
import torch
from ultralytics import YOLO


def train_model(
    data_yaml,
    base_model="yolov8n.pt",
    epochs=50,
    batch_size=16,
    img_size=640,
    device="auto",
    project="runs/detect",
    name="human_tracker_model",
):
    if not os.path.exists(data_yaml):
        print(f"Error: Dataset configuration file '{data_yaml}' not found.")
        return False

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"--- Starting YOLO Model Training ---")
    print(f"Base Model: {base_model}")
    print(f"Dataset Config: {data_yaml}")
    print(f"Epochs: {epochs} | Batch Size: {batch_size} | Image Size: {img_size}")
    print(f"Device: {device}")

    model = YOLO(base_model)
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        project=project,
        name=name,
        exist_ok=True,
        pretrained=True,
        verbose=True,
    )

    best_weights = os.path.join(project, name, "weights", "best.pt")
    if os.path.exists(best_weights):
        print(f"\nTraining completed successfully! Best model saved to: {best_weights}")
        return best_weights
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Custom YOLO Human Tracker Model")
    parser.add_argument("--data", type=str, required=True, help="Path to data.yaml dataset file")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Base model weights")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution size")
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda/cpu/auto)")
    args = parser.parse_args()

    train_model(
        data_yaml=args.data,
        base_model=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        img_size=args.imgsz,
        device=args.device,
    )
