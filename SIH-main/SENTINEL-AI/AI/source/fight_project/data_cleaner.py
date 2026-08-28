import os
import glob
import csv
import json
import numpy as np
import cv2
from collections import Counter
from sklearn.model_selection import train_test_split
from ultralytics import YOLO

def extract_middle_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None

def extract_group_id(filepath):
    """
    Extracts root source video ID to prevent train/val leakage.
    Example: 'train_0012_Fight_clip3.avi' -> 'train_0012'
    """
    basename = os.path.basename(filepath)
    parts = basename.split("_")
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return basename

def clean_and_split_dataset(data_dir="data", output_split_file="splits.json", csv_log="discarded_clips.csv"):
    """
    Cleans dataset clips and performs video-group-level 70% Train / 15% Val / 15% Test split
    to prevent frame/source leakage across splits.
    """
    print("Initializing YOLOv8 person detector for data cleaning pass...")
    detector = YOLO("yolov8n.pt")

    fight_files = glob.glob(os.path.join(data_dir, "Fight", "*.*"))
    non_fight_files = glob.glob(os.path.join(data_dir, "NonFight", "*.*"))

    all_files = [(f, "Fight") for f in fight_files] + [(f, "NonFight") for f in non_fight_files]
    print(f"Total candidate video clips found: {len(all_files)}")

    valid_samples = []
    discarded = []

    for video_path, label in all_files:
        filename = os.path.basename(video_path)
        mid_frame = extract_middle_frame(video_path)
        if mid_frame is None:
            discarded.append((filename, label, "Corrupted/Unreadable Video"))
            continue

        results = detector(mid_frame, verbose=False, classes=[0])
        person_count = len(results[0].boxes) if len(results) > 0 else 0

        if person_count < 2:
            discarded.append((filename, label, f"Insufficient Persons ({person_count} < 2)"))
            continue

        valid_samples.append((video_path, label))

    if not valid_samples:
        print("Warning: No valid samples found! Using raw files for split as fallback.")
        valid_samples = all_files

    # Log discarded
    os.makedirs(os.path.dirname(csv_log) or ".", exist_ok=True)
    with open(csv_log, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Filename", "Label", "Reason"])
        for row in discarded:
            writer.writerow(row)

    # Stratified split by Group ID to prevent train/val leakage
    group_map = {}
    for path, label in valid_samples:
        gid = extract_group_id(path)
        if gid not in group_map:
            group_map[gid] = {"samples": [], "label": label}
        group_map[gid]["samples"].append({"path": path, "label": label})

    g_ids = list(group_map.keys())
    g_labels = [group_map[g]["label"] for g in g_ids]

    if len(g_ids) >= 3:
        g_train, g_temp, _, g_temp_labels = train_test_split(
            g_ids, g_labels, test_size=0.30, stratify=g_labels, random_state=42
        )
        g_val, g_test, _, _ = train_test_split(
            g_temp, g_temp_labels, test_size=0.50, stratify=g_temp_labels, random_state=42
        )

        train_samples = [s for g in g_train for s in group_map[g]["samples"]]
        val_samples = [s for g in g_val for s in group_map[g]["samples"]]
        test_samples = [s for g in g_test for s in group_map[g]["samples"]]
    else:
        train_samples = [s for path, label in valid_samples for s in [{"path": path, "label": label}]]
        val_samples = train_samples
        test_samples = train_samples

    splits = {
        "train": train_samples,
        "val": val_samples,
        "test": test_samples
    }

    with open(output_split_file, "w") as f:
        json.dump(splits, f, indent=4)

    print(f"Group-level split saved to '{output_split_file}':")
    print(f"  Train: {len(train_samples)} clips")
    print(f"  Val:   {len(val_samples)} clips")
    print(f"  Test:  {len(test_samples)} clips")

    return splits

if __name__ == "__main__":
    clean_and_split_dataset()
