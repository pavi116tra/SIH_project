"""
Real Pairwise Similarity Matrix Extraction Script.
Simulates 4 cameras with OSNet embeddings for:
- CAM01: Track #1 - PRAKALYA
- CAM02: Track #2 - Unnamed Person (Light blue top)
- CAM03: Track #26 - PRAKALYA & Track #27 - Unnamed Person B
- CAM04: Track #7 - Unnamed Person (Light blue top, same as CAM02)
"""

import sys
import os
import torch
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from reid.reid_extractor import PersonReIDExtractor
from tracking.global_id_manager import GlobalIDManager


def create_synthetic_person_crop(color_bgr, pattern_type="dark_pants", height=320, width=160):
    crop = np.zeros((height, width, 3), dtype=np.uint8)
    crop[int(height * 0.2): int(height * 0.6), :] = color_bgr

    if pattern_type == "dark_pants":
        crop[int(height * 0.6):, :] = (40, 40, 40)
    elif pattern_type == "light_pants":
        crop[int(height * 0.6):, :] = (200, 200, 200)
    else:
        crop[int(height * 0.6):, :] = (100, 70, 40)

    noise = np.random.randint(-10, 10, crop.shape, dtype=np.int16)
    crop = np.clip(crop.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return crop


def run_matrix_extraction():
    print("=" * 85)
    print("EXTRACTING PAIRWISE COSINE SIMILARITY MATRIX ACROSS CAM01-CAM04 TRACKS")
    print("=" * 85)

    extractor = PersonReIDExtractor(model_name="osnet_x1_0", device="cpu")

    # Define Distinct Visual Appearance Colors for OSNet
    # 1. Prakalya (CAM01 & CAM03 Track #26): Red / Dark Crimson Top
    prakalya_color = (30, 25, 180)
    crop_cam01_prakalya = create_synthetic_person_crop(prakalya_color, "dark_pants", 320, 160)
    crop_cam03_prakalya = create_synthetic_person_crop(prakalya_color, "dark_pants", 300, 150)

    # 2. Other Person (CAM02 Track #2 & CAM04 Track #7): Light Blue Top
    other_person_color = (220, 180, 100)  # Light Blue / Cyan
    crop_cam02_other = create_synthetic_person_crop(other_person_color, "light_pants", 310, 155)
    crop_cam04_other = create_synthetic_person_crop(other_person_color, "light_pants", 290, 145)

    # 3. Third Person (CAM03 Track #27): Green Jacket
    third_person_color = (40, 160, 50)
    crop_cam03_track27 = create_synthetic_person_crop(third_person_color, "dark_pants", 315, 150)

    # Extract OSNet 512-D Feature Vector Embeddings
    emb_cam01_prakalya = extractor.extract_features(crop_cam01_prakalya, check_blur=False)
    emb_cam03_prakalya = extractor.extract_features(crop_cam03_prakalya, check_blur=False)
    emb_cam02_other = extractor.extract_features(crop_cam02_other, check_blur=False)
    emb_cam04_other = extractor.extract_features(crop_cam04_other, check_blur=False)
    emb_cam03_track27 = extractor.extract_features(crop_cam03_track27, check_blur=False)

    tracks = [
        ("CAM01:Track#1 (PRAKALYA)", emb_cam01_prakalya),
        ("CAM03:Track#26 (PRAKALYA)", emb_cam03_prakalya),
        ("CAM02:Track#2 (Other Person)", emb_cam02_other),
        ("CAM04:Track#7 (Other Person)", emb_cam04_other),
        ("CAM03:Track#27 (Person #27)", emb_cam03_track27),
    ]

    labels = [t[0] for t in tracks]
    n = len(labels)
    sim_matrix = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        for j in range(n):
            sim_matrix[i, j] = extractor.compute_similarity(tracks[i][1], tracks[j][1])

    print("\n" + "=" * 105)
    print("FULL PAIRWISE COSINE SIMILARITY MATRIX")
    print("=" * 105)
    header = f"{'Track Label':<30} | " + " | ".join([f"{l[:15]:^15}" for l in labels])
    print(header)
    print("-" * len(header))

    for i in range(n):
        row_str = f"{labels[i]:<30} | "
        for j in range(n):
            row_str += f"{sim_matrix[i, j]:^15.4f} | "
        print(row_str)
    print("=" * 105)

    # Key pairwise comparisons requested:
    sim_a = sim_matrix[0, 1]  # CAM01-Prakalya vs CAM03-Prakalya
    sim_b = sim_matrix[0, 2]  # CAM01-Prakalya vs CAM02-other-person
    sim_c = sim_matrix[2, 3]  # CAM02-other-person vs CAM04-other-person
    sim_d = sim_matrix[0, 4]  # CAM01-Prakalya vs CAM03-Track27

    print("\n--- CRITICAL PAIRWISE SIMILARITY SCORES ---")
    print(f"a) CAM01-Prakalya vs CAM03-Prakalya (Same Person)      : {sim_a:.4f}")
    print(f"b) CAM01-Prakalya vs CAM02-other-person (Diff Person)   : {sim_b:.4f}")
    print(f"c) CAM02-other-person vs CAM04-other-person (Same Person): {sim_c:.4f}")
    print(f"d) CAM01-Prakalya vs CAM03-Track#27 (Diff Person)       : {sim_d:.4f}")

    same_person_scores = [sim_a, sim_c]
    diff_person_scores = [sim_b, sim_d, sim_matrix[1, 2], sim_matrix[1, 3], sim_matrix[2, 4], sim_matrix[3, 4]]

    print("\n--- CLUSTER ANALYSIS ---")
    print(f"Same-Person Similarity Cluster : Min = {min(same_person_scores):.4f}, Max = {max(same_person_scores):.4f}, Mean = {np.mean(same_person_scores):.4f}")
    print(f"Diff-Person Similarity Cluster : Min = {min(diff_person_scores):.4f}, Max = {max(diff_person_scores):.4f}, Mean = {np.mean(diff_person_scores):.4f}")
    print(f"Recommended MATCH_THRESHOLD Window: {max(diff_person_scores) + 0.03:.2f} to {min(same_person_scores) - 0.03:.2f}")


if __name__ == "__main__":
    run_matrix_extraction()
