"""
Unit Test for Panda & Wildlife Phone Screen Photo Classification.
Verifies that ZeroShotClassifier correctly classifies Panda / Giant Panda crops.
"""

import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.zero_shot_classifier import ZeroShotClassifier


def create_synthetic_panda_crop():
    """
    Creates a black and white panda pattern test crop.
    """
    img = np.full((200, 200, 3), 240, dtype=np.uint8) # White body background
    # Black patches for eyes & ears
    cv2.circle(img, (50, 40), 25, (20, 20, 20), -1)  # Left ear
    cv2.circle(img, (150, 40), 25, (20, 20, 20), -1) # Right ear
    cv2.ellipse(img, (70, 90), (20, 15), 30, 0, 360, (20, 20, 20), -1)  # Left eye patch
    cv2.ellipse(img, (130, 90), (20, 15), -30, 0, 360, (20, 20, 20), -1) # Right eye patch
    cv2.ellipse(img, (100, 120), (12, 8), 0, 0, 360, (10, 10, 10), -1)   # Nose
    cv2.putText(img, "PANDA", (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    return img


def test_panda_classification():
    print("=== STARTING PANDA & WILDLIFE ZERO-SHOT TEST ===")
    classifier = ZeroShotClassifier(model_name="openai/clip-vit-base-patch32")
    assert classifier.is_ready, "Classifier not initialized!"

    crop = create_synthetic_panda_crop()
    refined_label, confidence = classifier.classify_crop(crop)

    print(f"Detected Label : {refined_label}")
    print(f"Confidence     : {confidence * 100:.2f}%")

    assert refined_label in ["panda", "giant panda", "bear"], f"Unexpected label: {refined_label}"
    print("=== PANDA ZERO-SHOT CLASSIFICATION TEST PASSED! ===")


if __name__ == "__main__":
    test_panda_classification()
