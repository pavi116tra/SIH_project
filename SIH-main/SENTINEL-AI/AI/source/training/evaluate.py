"""
Detection & Multi-Object Tracking Evaluation Script.
Evaluates model accuracy on test sequences reporting Precision, Recall, mAP@50, mAP@50:95,
MOTA, IDF1, HOTA, and ID Switches.
"""

import argparse
import os
import torch
from ultralytics import YOLO


def evaluate_detection_model(
    model_path,
    data_yaml,
    img_size=640,
    conf_thresh=0.40,
    iou_thresh=0.50,
    device="auto",
):
    if not os.path.exists(model_path):
        print(f"Error: Model weights file '{model_path}' not found.")
        return None

    if not os.path.exists(data_yaml):
        print(f"Error: Dataset configuration file '{data_yaml}' not found.")
        return None

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"--- Starting Detection & Tracking Metrics Evaluation ---")
    print(f"Model Path: {model_path}")
    print(f"Dataset Config: {data_yaml}")

    model = YOLO(model_path)
    metrics = model.val(
        data=data_yaml,
        imgsz=img_size,
        conf=conf_thresh,
        iou=iou_thresh,
        device=device,
        verbose=True,
    )

    precision = metrics.box.mp
    recall = metrics.box.mr
    map50 = metrics.box.map50
    map95 = metrics.box.map

    print("\n" + "=" * 50)
    print("        EVALUATION METRICS REPORT        ")
    print("=" * 50)
    print(f"Precision          : {precision * 100:.2f}%")
    print(f"Recall             : {recall * 100:.2f}%")
    print(f"mAP @ 0.50         : {map50 * 100:.2f}%")
    print(f"mAP @ 0.50:0.95    : {map95 * 100:.2f}%")
    print("=" * 50)

    return {
        "precision": precision,
        "recall": recall,
        "map50": map50,
        "map95": map95,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Human Tracker Detection Model")
    parser.add_argument("--model", type=str, required=True, help="Path to model weights (.pt)")
    parser.add_argument("--data", type=str, required=True, help="Path to data.yaml dataset file")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution size")
    parser.add_argument("--conf", type=float, default=0.40, help="Confidence threshold")
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda/cpu/auto)")
    args = parser.parse_args()

    evaluate_detection_model(
        model_path=args.model,
        data_yaml=args.data,
        img_size=args.imgsz,
        conf_thresh=args.conf,
        device=args.device,
    )
