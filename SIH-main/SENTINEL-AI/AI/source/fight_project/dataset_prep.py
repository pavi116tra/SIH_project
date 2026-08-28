import os
import tarfile
import urllib.request
import cv2
import numpy as np
from collections import Counter

def crop_video_to_3_seconds(input_path, output_path, target_duration_sec=3.0):
    """
    Extracts a 3-second center portion from a 5-second video clip.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return False

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        return False

    target_frames = int(fps * target_duration_sec)

    if total_frames > target_frames:
        start_frame = (total_frames - target_frames) // 2
        end_frame = start_frame + target_frames
    else:
        start_frame = 0
        end_frame = total_frames

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')


    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current_frame = start_frame

    while cap.isOpened() and current_frame < end_frame:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        writer.write(frame)
        current_frame += 1

    cap.release()
    writer.release()
    return True

def extract_rwf2000_dataset(output_dir="data", max_clips_per_class=40):
    fight_dir = os.path.join(output_dir, "Fight")
    non_fight_dir = os.path.join(output_dir, "NonFight")
    os.makedirs(fight_dir, exist_ok=True)
    os.makedirs(non_fight_dir, exist_ok=True)

    url = "https://huggingface.co/datasets/DanJoshua/RWF-2000/resolve/main/RWF-2000.tar.gz"
    print(f"Streaming real RWF-2000 surveillance dataset from HuggingFace:\n  {url}", flush=True)

    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    response = urllib.request.urlopen(req)

    tf = tarfile.open(fileobj=response, mode="r|gz")
    class_counts = Counter()

    temp_raw_path = os.path.join(output_dir, "temp_raw.avi")

    for member in tf:
        if not member.isfile() or not member.name.endswith(".avi"):
            continue

        name_lower = member.name.lower()
        if "fight" in name_lower and "nonfight" not in name_lower and "non_fight" not in name_lower and "non-fight" not in name_lower:
            label = "Fight"
        elif "nonfight" in name_lower or "non_fight" in name_lower or "non-fight" in name_lower:
            label = "NonFight"
        else:
            continue

        if class_counts[label] >= max_clips_per_class:
            if class_counts["Fight"] >= max_clips_per_class and class_counts["NonFight"] >= max_clips_per_class:
                break
            continue

        f_obj = tf.extractfile(member)
        if f_obj is None:
            continue

        with open(temp_raw_path, "wb") as f_out:
            f_out.write(f_obj.read())

        filename = f"rwf2000_{class_counts[label]:04d}_{label}.avi"
        target_dir = fight_dir if label == "Fight" else non_fight_dir
        final_path = os.path.join(target_dir, filename)

        success = crop_video_to_3_seconds(temp_raw_path, final_path, target_duration_sec=3.0)
        if success:
            class_counts[label] += 1
            print(f"  [RWF-2000] Extracted 3s clip #{class_counts[label]} ({label}): {filename}", flush=True)

    if os.path.exists(temp_raw_path):
        os.remove(temp_raw_path)

    print(f"\n--- Real RWF-2000 Dataset Extraction Complete ---", flush=True)
    print(f"Total 3-second clips saved: {sum(class_counts.values())}", flush=True)
    print(f"Class Distribution: {dict(class_counts)}", flush=True)
    return True

if __name__ == "__main__":
    extract_rwf2000_dataset(max_clips_per_class=40)
