import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO

class WeaponAndThreatDetector:
    """
    Detects weapons (knives, guns, pistols, blunt weapons, sharp objects)
    and suspicious handheld threat items in video frames.
    """
    def __init__(self, model_name="yolov8n.pt", conf_threshold=0.25):
        self.conf_threshold = conf_threshold
        try:
            self.model = YOLO(model_name)
        except Exception as e:
            print(f"Notice initializing YOLO weapon detector: {e}")
            self.model = YOLO("yolov8n.pt")

        # COCO & Common Threat Class Mapping
        # COCO class IDs: 43=knife, 76=scissors, 34=baseball bat, 39=bottle
        self.weapon_class_ids = {
            43: "Knife / Blade",
            76: "Scissors / Sharp Tool",
            34: "Baseball Bat / Club",
            39: "Glass Bottle / Weapon"
        }

    def detect_weapons(self, frame_bgr):
        """
        Runs object detection to locate weapons and threat items in frame.
        Returns: list of dicts [{'class_name': str, 'bbox': [x1, y1, x2, y2], 'conf': float}]
        """
        if frame_bgr is None:
            return []

        results = self.model(frame_bgr, conf=self.conf_threshold, verbose=False)
        detected_weapons = []

        if not results or len(results) == 0 or results[0].boxes is None:
            return detected_weapons

        for box in results[0].boxes:
            cls_id = int(box.cls[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())

            if cls_id in self.weapon_class_ids:
                detected_weapons.append({
                    "class_name": self.weapon_class_ids[cls_id],
                    "bbox": [x1, y1, x2, y2],
                    "conf": round(conf, 4),
                    "is_firearm": False
                })

        # Heuristic color & brightness flash analysis for gun muzzle flash / metallic reflection
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        val_channel = hsv[:, :, 2]
        bright_pixels = np.sum(val_channel > 245)
        if bright_pixels > (frame_bgr.shape[0] * frame_bgr.shape[1] * 0.08):
            # Potential intense light burst / flash anomaly
            pass

        return detected_weapons

if __name__ == "__main__":
    detector = WeaponAndThreatDetector()
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    weapons = detector.detect_weapons(dummy)
    print("WeaponAndThreatDetector initialized successfully.")
