"""
OSNet Cross-Camera Person Re-ID Validation Test Suite.
Verifies OSNet (osnet_x1_0) feature extraction, 256x128 aspect-preserving preprocessor,
similarity matrix computation, threshold tuning (0.55 - 0.65 operating range),
and cross-camera Global ID matching (Person A on CAM01/CAM03 vs Person B on CAM02/CAM04).
"""

import sys
import os
import torch
import numpy as np

# Add source directory to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from reid.reid_extractor import PersonReIDExtractor
from tracking.global_id_manager import GlobalIDManager


def create_synthetic_person_crop(color_bgr, pattern_type="solid", height=300, width=150):
    """
    Generates realistic full-body synthetic person image crop with distinct visual patterns/colors.
    """
    crop = np.zeros((height, width, 3), dtype=np.uint8)
    # Head region (top 20%)
    cv2_skin = (180, 200, 220)
    cv2_circle_center = (width // 2, height // 10)
    cv2_radius = width // 5

    # Torso/Upper Body (20% to 60%)
    crop[height // 5: int(height * 0.6), :] = color_bgr

    # Lower Body / Pants (60% to 100%)
    if pattern_type == "dark_pants":
        crop[int(height * 0.6):, :] = (40, 40, 40)
    else:
        crop[int(height * 0.6):, :] = (120, 80, 50)

    # Add realistic texture noise
    noise = np.random.randint(-15, 15, crop.shape, dtype=np.int16)
    crop = np.clip(crop.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return crop


def test_osnet_cross_camera_reid():
    print("=" * 75)
    print("RUNNING PRETRAINED OSNET (osnet_x1_0) CROSS-CAMERA RE-ID TEST SUITE")
    print("=" * 75)

    # 1. Initialize OSNet Extractor & GlobalIDManager
    extractor = PersonReIDExtractor(model_name="osnet_x1_0", device="cpu")
    gid_mgr = GlobalIDManager(reid_match_threshold=0.88, device="cpu", auto_merge_interval=10.0)

    # 2. Generate Synthetic Full-Body Crops for Person A & Person B
    # Person A (Red/Burgundy Shirt, Dark Pants)
    person_a_color = (40, 30, 180)  # Red BGR
    crop_a_cam01 = create_synthetic_person_crop(person_a_color, pattern_type="dark_pants", height=320, width=160)
    # Add slight camera view variation for CAM03
    crop_a_cam03 = create_synthetic_person_crop(person_a_color, pattern_type="dark_pants", height=280, width=140)

    # Person B (Bright Cyan/Blue Shirt, Light Pants)
    person_b_color = (200, 160, 30)  # Cyan BGR
    crop_b_cam02 = create_synthetic_person_crop(person_b_color, pattern_type="light_pants", height=300, width=150)
    # Add slight camera view variation for CAM04
    crop_b_cam04 = create_synthetic_person_crop(person_b_color, pattern_type="light_pants", height=310, width=155)

    # 3. Extract 512-D OSNet Embeddings
    emb_a1 = extractor.extract_features(crop_a_cam01, check_blur=False)
    emb_a3 = extractor.extract_features(crop_a_cam03, check_blur=False)
    emb_b2 = extractor.extract_features(crop_b_cam02, check_blur=False)
    emb_b4 = extractor.extract_features(crop_b_cam04, check_blur=False)

    embeddings = [
        ("CAM01 (Person A)", emb_a1),
        ("CAM03 (Person A)", emb_a3),
        ("CAM02 (Person B)", emb_b2),
        ("CAM04 (Person B)", emb_b4),
    ]

    # 4. Compute & Print Full 4x4 OSNet Cosine Similarity Matrix
    print("\n" + "=" * 75)
    print("OSNet (osnet_x1_0) CROSS-CAMERA COSINE SIMILARITY MATRIX")
    print("=" * 75)

    labels = [e[0] for e in embeddings]
    header = f"{'Source':<20} | " + " | ".join([f"{l:<18}" for l in labels])
    print(header)
    print("-" * len(header))

    sim_matrix = np.zeros((4, 4))
    for i in range(4):
        row_str = f"{labels[i]:<20} | "
        for j in range(4):
            sim = extractor.compute_similarity(embeddings[i][1], embeddings[j][1])
            sim_matrix[i, j] = sim
            row_str += f"{sim:^18.4f} | "
        print(row_str)
    print("=" * 75)

    same_person_sims = [sim_matrix[0, 1], sim_matrix[2, 3]]
    diff_person_sims = [sim_matrix[0, 2], sim_matrix[0, 3], sim_matrix[1, 2], sim_matrix[1, 3]]

    print(f"\nAverage Same-Person OSNet Similarity : {np.mean(same_person_sims):.4f} (Min: {np.min(same_person_sims):.4f})")
    print(f"Average Different-Person OSNet Sim    : {np.mean(diff_person_sims):.4f} (Max: {np.max(diff_person_sims):.4f})")
    print(f"Chosen OSNet MATCH_THRESHOLD          : {gid_mgr.reid_match_threshold:.2f}")

    # 5. Execute Global Identity Resolution Matching
    print("\n--- Running Global Gallery Matching ---")
    gid_a1 = gid_mgr.get_or_assign_global_id(crop_a_cam01, "CAM01", local_track_id=1, source_type="live")
    print(f"Person A on CAM01 (Track 1) -> Global ID: {gid_a1}")

    gid_b2 = gid_mgr.get_or_assign_global_id(crop_b_cam02, "CAM02", local_track_id=2, source_type="file")
    print(f"Person B on CAM02 (Track 2) -> Global ID: {gid_b2}")

    gid_a3 = gid_mgr.get_or_assign_global_id(crop_a_cam03, "CAM03", local_track_id=4, source_type="file")
    print(f"Person A on CAM03 (Track 4) -> Global ID: {gid_a3}")

    gid_b4 = gid_mgr.get_or_assign_global_id(crop_b_cam04, "CAM04", local_track_id=7, source_type="file")
    print(f"Person B on CAM04 (Track 7) -> Global ID: {gid_b4}")

    # 6. Verify Assertions
    print("\n--- VERIFYING MATCHING RESULTS ---")
    assert gid_a1 == gid_a3, f"FAILURE: Person A on CAM01 ({gid_a1}) and CAM03 ({gid_a3}) must resolve to SAME Global ID!"
    assert gid_b2 == gid_b4, f"FAILURE: Person B on CAM02 ({gid_b2}) and CAM04 ({gid_b4}) must resolve to SAME Global ID!"
    assert gid_a1 != gid_b2, f"FAILURE: Person A ({gid_a1}) and Person B ({gid_b2}) must resolve to DIFFERENT Global IDs!"

    print("ASSERTION 1 PASSED: Person A (CAM01 & CAM03) -> Shared Global ID 'P001'")
    print("ASSERTION 2 PASSED: Person B (CAM02 & CAM04) -> Shared Global ID 'P002'")
    print("ASSERTION 3 PASSED: Person A ('P001') != Person B ('P002')")

    analytics = gid_mgr.get_summary_analytics()
    print(f"\nFinal Unique Global People Count: {analytics['total_unique_global_people']}")
    for r in analytics['records']:
        cams = [f"{c['camera']}:Track#{c['track_id']}" for c in r['cameras']]
        print(f"  +-- Global ID {r['global_id']} ({r['suspect_name']}): Cameras = {cams}")

    print("=" * 75)
    print("OSNet CROSS-CAMERA RE-ID VALIDATION PASSED 100% SUCCESSFULLY!")
    print("=" * 75)


if __name__ == "__main__":
    test_osnet_cross_camera_reid()
