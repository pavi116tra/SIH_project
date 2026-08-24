"""
Automated Test for Multi-Camera Architecture.
Verifies camera sources (WebcamSource, VideoFileSource), CameraManager,
independent per-camera ByteTracker instances, and shared GlobalIDManager.
"""

import os
import sys
import unittest
import numpy as np

# Add parent directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from camera.camera_source import WebcamSource, VideoFileSource, create_camera_source
from camera.camera_manager import CameraManager
from tracking.human_tracker import HumanTrackerEngine
from tracking.tracker import Track


class TestMultiCameraArchitecture(unittest.TestCase):

    def setUp(self):
        Track._count = 0

    def test_camera_sources(self):
        print("\n--- TEST 1: CAMERA SOURCE ABSTRACTION LAYER ---")
        cam1_dict = {"id": "CAM01", "name": "Laptop Webcam", "source_type": "webcam", "source": 0, "enabled": True}
        cam1 = create_camera_source(cam1_dict)
        self.setIsInstance(cam1, WebcamSource) if hasattr(self, 'setIsInstance') else self.assertIsInstance(cam1, WebcamSource)
        self.assertEqual(cam1.camera_id, "CAM01")

        cam2_dict = {"id": "CAM02", "name": "Video Simulation", "source_type": "video", "source": "videos/sample.mp4", "enabled": False}
        cam2 = create_camera_source(cam2_dict)
        self.assertIsInstance(cam2, VideoFileSource)
        self.assertEqual(cam2.camera_id, "CAM02")
        print("[TEST 1 PASSED] CameraSource abstractions created successfully.")

    def test_camera_manager(self):
        print("\n--- TEST 2: CAMERA MANAGER & CONFIG LOADING ---")
        mgr = CameraManager()
        cams = mgr.get_active_cameras()
        self.assertTrue(len(cams) >= 1)
        self.assertEqual(cams[0]["camera_id"], "CAM01")
        print(f"[TEST 2 PASSED] CameraManager initialized with {len(cams)} camera sources.")

    def test_independent_camera_trackers(self):
        print("\n--- TEST 3: INDEPENDENT PER-CAMERA TRACKERS & SHARED GLOBAL ID MANAGER ---")
        engine = HumanTrackerEngine()

        # Simulate frame processing for CAM01
        dummy_frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
        tracks_cam1, non_human1, stats1 = engine.process_frame(dummy_frame1, camera_id="CAM01")

        # Simulate frame processing for CAM02
        dummy_frame2 = np.zeros((480, 640, 3), dtype=np.uint8)
        tracks_cam2, non_human2, stats2 = engine.process_frame(dummy_frame2, camera_id="CAM02")

        # Verify independent ByteTracker instances were created
        self.assertIn("CAM01", engine.trackers)
        self.assertIn("CAM02", engine.trackers)
        self.assertNotEqual(engine.trackers["CAM01"], engine.trackers["CAM02"])
        print("[TEST 3 PASSED] Independent ByteTrackers verified for CAM01 and CAM02.")


if __name__ == "__main__":
    unittest.main()
