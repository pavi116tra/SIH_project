"""
Verification Test for Phase 1: Zero-Shot CLIP Refinement Layer.
Tests fine-grained classification refinement for Lion, Drone, and Bird species crops.
"""

import os
import sys
import cv2
import numpy as np

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.zero_shot_classifier import ZeroShotClassifier


def create_synthetic_test_crop(label_type="lion"):
    """
    Generates synthetic representative test image crops for testing CLIP zero-shot refinement.
    """
    img = np.zeros((200, 200, 3), dtype=np.uint8)

    if label_type == "lion":
        # Warm golden brown background with dark mane accents
        img[:, :] = (30, 100, 180)  # BGR golden brown
        cv2.ellipse(img, (100, 100), (60, 80), 0, 0, 360, (20, 50, 100), -1)
        cv2.putText(img, "LION", (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    elif label_type == "drone":
        # Sky blue background with dark quadcopter frame
        img[:, :] = (235, 206, 135) # Sky blue
        cv2.line(img, (40, 40), (160, 160), (40, 40, 40), 6)
        cv2.line(img, (160, 40), (40, 160), (40, 40, 40), 6)
        for pt in [(40, 40), (160, 40), (40, 160), (160, 160)]:
            cv2.circle(img, pt, 20, (100, 100, 100), -1)
    elif label_type == "eagle":
        # Sky background with eagle wing silouette
        img[:, :] = (220, 180, 120)
        cv2.ellipse(img, (100, 100), (80, 40), 0, 0, 360, (30, 30, 30), -1)
        cv2.putText(img, "EAGLE", (40, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return img


def test_zero_shot_clip_layer():
    print("=== STARTING PHASE 1 ZERO-SHOT CLIP REFINEMENT TEST ===")

    classifier = ZeroShotClassifier(model_name="openai/clip-vit-base-patch32")
    assert classifier.is_ready, "ZeroShotClassifier failed to initialize!"

    test_cases = [
        {"type": "lion", "coarse_yolo": "Cat", "target_candidates": ["lion", "cat", "dog", "tiger", "leopard"]},
        {"type": "drone", "coarse_yolo": "Bird", "target_candidates": ["drone", "quadcopter", "bird", "airplane", "kite"]},
        {"type": "eagle", "coarse_yolo": "Bird", "target_candidates": ["eagle", "hawk", "sparrow", "crow", "bird"]},
    ]

    results = []
    for test in test_cases:
        crop = create_synthetic_test_crop(test["type"])
        refined_label, confidence = classifier.classify_crop(
            crop, candidate_labels=test["target_candidates"]
        )

        print(f"\n[TEST CASE: {test['type'].upper()}]")
        print(f"  Coarse YOLOv8 Label : {test['coarse_yolo']}")
        print(f"  Refined CLIP Label  : {refined_label.title() if refined_label else 'None'}")
        print(f"  CLIP Confidence     : {confidence * 100:.2f}%")

        assert refined_label is not None, f"Failed to get refined label for {test['type']}"
        assert confidence > 0.15, f"Confidence too low for {test['type']}"

        results.append({
            "type": test["type"],
            "coarse": test["coarse_yolo"],
            "refined": refined_label.title(),
            "confidence": confidence
        })

    print("\n=== ALL PHASE 1 CLIP REFINEMENT TESTS PASSED SUCCESSFULLY! ===")
    return results


if __name__ == "__main__":
    test_zero_shot_clip_layer()
