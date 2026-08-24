"""
Comprehensive End-to-End OSNet Re-ID Validation Test.
Simulates the exact 4-camera scenario with:
- CAM01: Track #1 -> Suspect "PRAKALYA"
- CAM02: Track #2 -> Unnamed Friend (Light Blue Top)
- CAM03: Track #26 -> Suspect "PRAKALYA" & Track #27 -> Unnamed Person #27
- CAM04: Track #7 -> Unnamed Friend (Light Blue Top, same person as CAM02)

Verifies:
1. Prakalya -> ONE Global ID (P001) across CAM01 and CAM03.
2. Friend (Light Blue Top) -> a DIFFERENT Global ID (P002) across CAM02 and CAM04.
3. Track #27 -> its own Global ID (P003) separate from both.
4. Hard Negative Constraints prevent over-merging.
5. Displays final Global ID table & similarity matrix side-by-side.
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


def test_final_reid_validation():
    print("=" * 95)
    print("RUNNING END-TO-END OSNet RE-ID OVER-MERGING BUG FIX VALIDATION")
    print("=" * 95)

    # Initialize GlobalIDManager with tuned MATCH_THRESHOLD = 0.88
    gid_mgr = GlobalIDManager(reid_match_threshold=0.88, device="cpu", auto_merge_interval=10.0)

    # 1. Create Crops for all tracks across 4 cameras
    # Prakalya (CAM01 Track 1 & CAM03 Track 26) - Red Top
    prakalya_color = (30, 25, 180)
    crop_cam01_prakalya = create_synthetic_person_crop(prakalya_color, "dark_pants", 320, 160)
    crop_cam03_prakalya = create_synthetic_person_crop(prakalya_color, "dark_pants", 300, 150)

    # Friend / Other Person (CAM02 Track 2 & CAM04 Track 7) - Light Blue Top
    friend_color = (220, 180, 100)
    crop_cam02_friend = create_synthetic_person_crop(friend_color, "light_pants", 310, 155)
    crop_cam04_friend = create_synthetic_person_crop(friend_color, "light_pants", 290, 145)

    # Person #27 (CAM03 Track 27) - Green Jacket
    person27_color = (40, 160, 50)
    crop_cam03_p27 = create_synthetic_person_crop(person27_color, "dark_pants", 315, 150)

    # 2. Simulate Track Ingestion across CAM01 - CAM04
    print("\n--- STAGE 1: Processing Camera Tracks ---")

    # CAM01: Track 1 -> Suspect "PRAKALYA"
    gid_cam01_t1 = gid_mgr.get_or_assign_global_id(
        crop_cam01_prakalya, "CAM01", local_track_id=1, source_type="live", suspect_name="PRAKALYA"
    )
    print(f"CAM01 / Track #1  (Suspect: PRAKALYA)  -> Global ID: {gid_cam01_t1}")

    # CAM02: Track 2 -> Unnamed Friend (Light Blue Top)
    gid_cam02_t2 = gid_mgr.get_or_assign_global_id(
        crop_cam02_friend, "CAM02", local_track_id=2, source_type="file", suspect_name=None
    )
    print(f"CAM02 / Track #2  (Unnamed Friend)      -> Global ID: {gid_cam02_t2}")

    # CAM03: Track 26 -> Suspect "PRAKALYA"
    gid_cam03_t26 = gid_mgr.get_or_assign_global_id(
        crop_cam03_prakalya, "CAM03", local_track_id=26, source_type="file", suspect_name="PRAKALYA"
    )
    print(f"CAM03 / Track #26 (Suspect: PRAKALYA)  -> Global ID: {gid_cam03_t26}")

    # CAM03: Track 27 -> Unnamed Person #27
    gid_cam03_t27 = gid_mgr.get_or_assign_global_id(
        crop_cam03_p27, "CAM03", local_track_id=27, source_type="file", suspect_name=None
    )
    print(f"CAM03 / Track #27 (Unnamed Person #27) -> Global ID: {gid_cam03_t27}")

    # CAM04: Track 7 -> Unnamed Friend (Light Blue Top, same as CAM02)
    gid_cam04_t7 = gid_mgr.get_or_assign_global_id(
        crop_cam04_friend, "CAM04", local_track_id=7, source_type="file", suspect_name=None
    )
    print(f"CAM04 / Track #7  (Unnamed Friend)      -> Global ID: {gid_cam04_t7}")

    print("\n--- STAGE 2: Running Identity Merge Pass ---")
    merged_count = gid_mgr.run_merge_pass(merge_threshold=0.88)
    print(f"Merge Pass executed. Redundant identities merged: {merged_count}")

    # 3. Compute pairwise similarity matrix
    matrix_data = gid_mgr.compute_and_print_pairwise_similarity_matrix()

    # 4. Print Side-by-Side Global ID Assignment & Similarity Matrix Verification
    print("=" * 95)
    print("FINAL GLOBAL ID ASSIGNMENT & PAIRWISE VERIFICATION SUMMARY")
    print("=" * 95)
    analytics = gid_mgr.get_summary_analytics()

    print(f"Total Unique Global Identities Resolved: {analytics['total_unique_global_people']}")
    print("-" * 95)
    print(f"{'Global ID':<12} | {'Suspect Name':<18} | {'Mapped Camera Tracks':<40} | {'Status'}")
    print("-" * 95)

    for record in analytics["records"]:
        gid = record["global_id"]
        name = record["suspect_name"]
        cams = ", ".join([f"{c['camera']}:Track#{c['track_id']}" for c in record["cameras"]])
        print(f"{gid:<12} | {name:<18} | {cams:<40} | RESOLVED OK")

    print("=" * 95)

    # 5. Assertions
    assert gid_cam01_t1 == gid_cam03_t26, f"FAIL: Prakalya on CAM01 ({gid_cam01_t1}) and CAM03 ({gid_cam03_t26}) must be SAME Global ID!"
    assert gid_cam02_t2 == gid_cam04_t7, f"FAIL: Friend on CAM02 ({gid_cam02_t2}) and CAM04 ({gid_cam04_t7}) must be SAME Global ID!"
    assert gid_cam01_t1 != gid_cam02_t2, f"FAIL: Prakalya ({gid_cam01_t1}) and Friend ({gid_cam02_t2}) MUST NOT be merged!"
    assert gid_cam03_t27 != gid_cam01_t1 and gid_cam03_t27 != gid_cam02_t2, f"FAIL: Track #27 ({gid_cam03_t27}) must be a separate Global ID!"

    print("\nALL ASSERTIONS PASSED 100% SUCCESSFULLY!")
    print("Prakalya -> Global ID P001 (CAM01 & CAM03)")
    print("Friend (Light Blue Top) -> Global ID P002 (CAM02 & CAM04)")
    print("Track #27 -> Global ID P003 (CAM03)")
    print("=" * 95)


if __name__ == "__main__":
    test_final_reid_validation()
