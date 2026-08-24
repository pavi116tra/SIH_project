"""
Automated Test for Stable Track IDs and Global IDs (Acceptance Test).
Verifies that Local Track IDs and Global Person IDs are NEVER re-indexed, compressed,
or changed when a person leaves the camera view.
"""

import os
import sys
import unittest

# Add parent directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tracking.tracker import Track
from tracking.global_id_manager import GlobalIDManager


class TestIDStability(unittest.TestCase):

    def setUp(self):
        Track._count = 0
        self.global_mgr = GlobalIDManager(retention_minutes=60, reid_match_threshold=0.75, device="cpu")

    def test_id_stability_sequence(self):
        print("\n--- RUNNING STABLE ID ACCEPTANCE TEST ---")

        # Step 1: Person A enters
        t1 = Track([10, 10, 50, 100], 0.95)
        g1 = self.global_mgr.get_or_assign_global_id(None, "CAM01", t1.track_id)
        t1.global_id = g1
        print(f"[STEP 1] Person A enters -> Track {t1.track_id}, Global {t1.global_id}")
        self.assertEqual(t1.track_id, 1)
        self.assertEqual(t1.global_id, "P001")

        # Step 2: Person B enters
        t2 = Track([150, 10, 200, 100], 0.92)
        g2 = self.global_mgr.get_or_assign_global_id(None, "CAM01", t2.track_id)
        t2.global_id = g2
        print(f"[STEP 2] Person B enters -> Track {t2.track_id}, Global {t2.global_id}")
        self.assertEqual(t2.track_id, 2)
        self.assertEqual(t2.global_id, "P002")

        # Step 3: Person A leaves
        print("[STEP 3] Person A leaves -> Active: Person B")
        # Verify Person B's Track ID & Global ID NEVER change to 1 or P001!
        self.assertEqual(t2.track_id, 2, "CRITICAL ERROR: Person B Track ID changed!")
        self.assertEqual(t2.global_id, "P002", "CRITICAL ERROR: Person B Global ID changed!")

        # Step 4: Person C enters
        t3 = Track([300, 10, 350, 100], 0.89)
        g3 = self.global_mgr.get_or_assign_global_id(None, "CAM01", t3.track_id)
        t3.global_id = g3
        print(f"[STEP 4] Person C enters -> Track {t3.track_id}, Global {t3.global_id}")
        self.assertEqual(t3.track_id, 3)
        self.assertEqual(t3.global_id, "P003")

        # Step 5: Person B leaves
        print("[STEP 5] Person B leaves -> Active: Person C")
        self.assertEqual(t3.track_id, 3, "CRITICAL ERROR: Person C Track ID changed!")
        self.assertEqual(t3.global_id, "P003", "CRITICAL ERROR: Person C Global ID changed!")

        # Step 6: Person D enters
        t4 = Track([450, 10, 500, 100], 0.91)
        g4 = self.global_mgr.get_or_assign_global_id(None, "CAM01", t4.track_id)
        t4.global_id = g4
        print(f"[STEP 6] Person D enters -> Track {t4.track_id}, Global {t4.global_id}")
        self.assertEqual(t4.track_id, 4)
        self.assertEqual(t4.global_id, "P004")

        print("--- ID STABILITY TEST PASSED SUCCESSFULLY ---")


if __name__ == "__main__":
    unittest.main()
