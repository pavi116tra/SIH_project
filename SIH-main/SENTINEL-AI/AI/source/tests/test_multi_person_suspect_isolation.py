"""
Multi-Person Suspect Isolation & Anti-Overmerging Regression Test Suite.
Tests co-occurrence of suspects and unregistered individuals in multi-camera streams,
verifying that appearance similarity gates temporary ID consolidation and prevents gallery pollution.
"""

import sys
import os
import numpy as np
import cv2

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tracking.global_id_manager import GlobalIDManager, GlobalPerson
from reid.reid_extractor import PersonReIDExtractor


def create_synthetic_person_crop(shirt_color_bgr, pants_type="dark_pants", height=300, width=150):
    """Generates synthetic full-body human crop with distinct shirt/pants color patterns."""
    crop = np.zeros((height, width, 3), dtype=np.uint8)
    head_h = int(height * 0.15)
    torso_h = int(height * 0.45)

    # Head (skin tone)
    crop[:head_h, :] = (180, 200, 230)

    # Torso (shirt color)
    crop[head_h:torso_h, :] = shirt_color_bgr

    # Pants
    if pants_type == "dark_pants":
        crop[torso_h:, :] = (30, 30, 30)
    else:
        crop[torso_h:, :] = (200, 200, 200)

    # Add realistic texture noise
    noise = np.random.randint(-10, 10, crop.shape, dtype=np.int16)
    crop = np.clip(crop.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return crop


def test_multi_person_suspect_isolation():
    print("=" * 95)
    print("RUNNING MULTI-PERSON SUSPECT ISOLATION & ANTI-OVERMERGING TEST SUITE")
    print("=" * 95)

    extractor = PersonReIDExtractor(model_name="osnet_x1_0", device="cpu")
    gid_mgr = GlobalIDManager(reid_match_threshold=0.88, device="cpu", auto_merge_interval=10.0, max_pending_wait_seconds=0.0)

    # 1. Create Crops for Suspect Pavitra S (Red Top) & Unregistered Friend (Cyan Top)
    red_shirt = (30, 25, 180)    # Pavitra S
    cyan_shirt = (220, 180, 30)  # Unregistered Friend
    green_shirt = (40, 160, 50)  # Unknown Person #3

    crop_pavitra_cam01 = create_synthetic_person_crop(red_shirt, "dark_pants", 320, 160)
    crop_pavitra_cam04 = create_synthetic_person_crop(red_shirt, "dark_pants", 300, 150)

    crop_friend_cam01 = create_synthetic_person_crop(cyan_shirt, "light_pants", 310, 155)
    crop_friend_cam02 = create_synthetic_person_crop(cyan_shirt, "light_pants", 295, 148)

    crop_green_cam03 = create_synthetic_person_crop(green_shirt, "dark_pants", 305, 150)

    # ---------------------------------------------------------------------------------------------
    # TEST 1: Simultaneous Multi-Person Frame Ingestion (Suspect + Unregistered Friend in CAM01)
    # ---------------------------------------------------------------------------------------------
    print("\n--- TEST 1: Simultaneous Multi-Person Ingestion in CAM01 ---")
    
    # Track 1: Face-anchored to suspect 'PAVITRA S'
    gid1 = gid_mgr.get_or_assign_global_id(crop_pavitra_cam01, "CAM01", local_track_id=1, suspect_name="PAVITRA S")
    print(f"CAM01 Track #1 (Suspect 'PAVITRA S') -> Assigned Global ID: {gid1}")
    assert gid1 == "P001", f"Expected P001, got {gid1}"

    # Track 2: Unregistered Friend in same frame (No face match)
    gid2 = gid_mgr.get_or_assign_global_id(crop_friend_cam01, "CAM01", local_track_id=2, suspect_name=None)
    print(f"CAM01 Track #2 (Unregistered Friend) -> Assigned Global ID: {gid2}")
    assert gid2 != "P001", f"FAILURE: Unregistered Friend in CAM01 was wrongfully merged into Suspect {gid1}!"
    assert gid2 == "P002", f"Expected P002 for Unregistered Friend, got {gid2}"

    # ---------------------------------------------------------------------------------------------
    # TEST 2: Cross-Camera Re-ID of Unregistered Friend (CAM02) vs Suspect (CAM04)
    # ---------------------------------------------------------------------------------------------
    print("\n--- TEST 2: Cross-Camera Re-ID of Friend on CAM02 vs Suspect on CAM04 ---")
    
    gid_friend_cam02 = gid_mgr.get_or_assign_global_id(crop_friend_cam02, "CAM02", local_track_id=5, suspect_name=None)
    print(f"CAM02 Track #5 (Friend appearance) -> Assigned Global ID: {gid_friend_cam02}")
    assert gid_friend_cam02 == "P002", f"FAILURE: Friend on CAM02 should resolve to P002, got {gid_friend_cam02}"

    gid_pavitra_cam04 = gid_mgr.get_or_assign_global_id(crop_pavitra_cam04, "CAM04", local_track_id=8, suspect_name="PAVITRA S")
    print(f"CAM04 Track #8 (Suspect 'PAVITRA S') -> Assigned Global ID: {gid_pavitra_cam04}")
    assert gid_pavitra_cam04 == "P001", f"FAILURE: Suspect on CAM04 should resolve to P001, got {gid_pavitra_cam04}"

    # ---------------------------------------------------------------------------------------------
    # TEST 3: Appearance-Gated Suspect Consolidation Rejection
    # ---------------------------------------------------------------------------------------------
    print("\n--- TEST 3: Temporary ID Consolidation Appearance Rejection ---")
    
    # Ingest Green Shirt Person on CAM03 Track 10 as temporary Global ID P003
    gid_temp = gid_mgr.get_or_assign_global_id(crop_green_cam03, "CAM03", local_track_id=10, suspect_name=None)
    print(f"CAM03 Track #10 (Green Shirt) -> Assigned Temporary Global ID: {gid_temp}")
    assert gid_temp == "P003", f"Expected P003, got {gid_temp}"

    # Now simulate a faulty face detection trigger claiming 'PAVITRA S' on CAM03 Track 10
    gid_consolidated = gid_mgr.get_or_assign_global_id(crop_green_cam03, "CAM03", local_track_id=10, suspect_name="PAVITRA S")
    print(f"CAM03 Track #10 after face match trigger -> Global ID: {gid_consolidated}")

    # Verify that P003 was NOT merged into P001 because green shirt appearance != Pavitra S red shirt!
    assert "P003" in gid_mgr.global_people, "FAILURE: Temporary ID P003 was wrongfully merged into P001 despite appearance mismatch!"
    print("SUCCESS: Appearance similarity gate rejected merging Green Shirt (P003) into Suspect P001!")

    # ---------------------------------------------------------------------------------------------
    # TEST 4: Prototype Gallery Crops Export & Diagnostic Inspection
    # ---------------------------------------------------------------------------------------------
    print("\n--- TEST 4: Exporting Prototype Gallery Crops ---")
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scratch", "test_p001_crops")
    saved = gid_mgr.export_prototype_gallery_crops(out_dir)
    print(f"Exported {len(saved)} prototype crop images to: {out_dir}")
    for item in saved:
        print(f"  +-- Prototype {item['global_id']} idx {item['prototype_idx']} saved at {item['filepath']}")

    print("\n" + "=" * 95)
    print("ALL MULTI-PERSON SUSPECT ISOLATION & CONSOLIDATION TESTS PASSED 100% SUCCESSFULLY!")
    print("=" * 95)


if __name__ == "__main__":
    test_multi_person_suspect_isolation()
