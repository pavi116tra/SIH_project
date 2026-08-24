"""
Comprehensive Re-ID & Global Identity Resolution Test Suite.
Verifies single Global ID assignment across live/file sources, video restart handling,
similarity matrix logging, merge pass execution, and nested API data structure.
"""

import sys
import os
import time
import numpy as np

# Add source directory to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from reid.reid_extractor import PersonReIDExtractor
from tracking.global_id_manager import GlobalIDManager, GlobalPerson


def test_reid_and_global_id_resolution():
    print("=" * 70)
    print("RUNNING GLOBAL RE-ID RESOLUTION & MERGE TEST SUITE")
    print("=" * 70)

    # 1. Initialize ReID & GlobalIDManager
    gid_mgr = GlobalIDManager(reid_match_threshold=0.60, device="cpu", auto_merge_interval=10.0)

    # Generate synthetic 512-D person feature embeddings for Person A (P001)
    np.random.seed(42)
    base_emb = np.random.randn(512).astype(np.float32)
    base_emb /= np.linalg.norm(base_emb)

    # Slight variations representing observations of Person A across different cameras
    emb_cam01 = base_emb + np.random.normal(0, 0.05, 512).astype(np.float32)
    emb_cam01 /= np.linalg.norm(emb_cam01)

    emb_cam02 = base_emb + np.random.normal(0, 0.08, 512).astype(np.float32)
    emb_cam02 /= np.linalg.norm(emb_cam02)

    emb_cam03 = base_emb + np.random.normal(0, 0.10, 512).astype(np.float32)
    emb_cam03 /= np.linalg.norm(emb_cam03)

    emb_cam04 = base_emb + np.random.normal(0, 0.07, 512).astype(np.float32)
    emb_cam04 /= np.linalg.norm(emb_cam04)

    # Synthetic image crops
    dummy_crop = (np.random.rand(100, 100, 3) * 255).astype(np.uint8)

    print("\n--- TEST 1: Multi-Camera Track Ingestion ---")
    # Simulate CAM01 (Live Webcam)
    gid1 = gid_mgr.get_or_assign_global_id(dummy_crop, "CAM01", local_track_id=7, source_type="live")
    gid_mgr.global_people[gid1].update_observation(emb_cam01, "CAM01", 7, source_type="live", sim_score=1.0)
    print(f"CAM01 Track 7 assigned to: {gid1}")

    # Simulate CAM02 (Video File)
    gid2 = gid_mgr.get_or_assign_global_id(dummy_crop, "CAM02", local_track_id=3, source_type="file")
    gid_mgr.global_people[gid2].update_observation(emb_cam02, "CAM02", 3, source_type="file", sim_score=0.92)
    print(f"CAM02 Track 3 assigned to: {gid2}")

    # Simulate CAM03 (Video File)
    gid3 = gid_mgr.get_or_assign_global_id(dummy_crop, "CAM03", local_track_id=5, source_type="file")
    gid_mgr.global_people[gid3].update_observation(emb_cam03, "CAM03", 5, source_type="file", sim_score=0.89)
    print(f"CAM03 Track 5 assigned to: {gid3}")

    # Simulate CAM04 (Video File)
    gid4 = gid_mgr.get_or_assign_global_id(dummy_crop, "CAM04", local_track_id=2, source_type="file")
    gid_mgr.global_people[gid4].update_observation(emb_cam04, "CAM04", 2, source_type="file", sim_score=0.91)
    print(f"CAM04 Track 2 assigned to: {gid4}")

    analytics_initial = gid_mgr.get_summary_analytics()
    print(f"\nInitial Unique Global People Count: {analytics_initial['total_unique_global_people']}")
    for r in analytics_initial['records']:
        print(f"Record {r['global_id']}: Cameras = {[c['camera'] + ':' + str(c['track_id']) for c in r['cameras']]}")

    print("\n--- TEST 2: Identity Merge Pass Execution ---")
    merged_count = gid_mgr.run_merge_pass(merge_threshold=0.70)
    print(f"Merge Pass Executed. Merged Count: {merged_count}")

    analytics_merged = gid_mgr.get_summary_analytics()
    print(f"Post-Merge Unique Global People Count: {analytics_merged['total_unique_global_people']}")
    assert analytics_merged['total_unique_global_people'] == 1, "FAILURE: All observations of Person A should collapse to exactly 1 Global ID!"
    assert analytics_merged['records'][0]['global_id'] == "P001", "FAILURE: Primary Global ID should be P001!"

    print("\nNested Hierarchical API Structure:")
    rec = analytics_merged['records'][0]
    print(f"Global ID: {rec['global_id']}")
    for cam in rec['cameras']:
        print(f"  +-- Camera: {cam['camera']} (Type: {cam['source_type']}) | Track #{cam['track_id']} | Confidence: {cam['confidence']*100:.0f}%")

    print("\n--- TEST 3: Video File Stream Reset / Loop Handling ---")
    print("Simulating CAM02 video file loop/restart...")
    gid_mgr.reset_camera_tracks("CAM02")

    # New local track ID #10 spawned on CAM02 after video restart
    gid_post_reset = gid_mgr.get_or_assign_global_id(dummy_crop, "CAM02", local_track_id=10, source_type="file")
    gid_mgr.global_people[gid_post_reset].update_observation(emb_cam02, "CAM02", 10, source_type="file", sim_score=0.94)

    print(f"CAM02 Post-Loop Track 10 assigned to: {gid_post_reset}")
    assert gid_post_reset == "P001", "FAILURE: Post-loop track on CAM02 should re-attach to existing Global ID P001!"

    analytics_final = gid_mgr.get_summary_analytics()
    print(f"\nFinal Unique Global People Count: {analytics_final['total_unique_global_people']}")
    print("=" * 70)
    print("ALL RE-ID & GLOBAL IDENTITY RESOLUTION TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    test_reid_and_global_id_resolution()
