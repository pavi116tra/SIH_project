"""
Zero-Shot CLIP Classification Refinement Layer.
Refines coarse YOLOv8 detections (e.g. Cat -> Lion, Bird/Kite -> Drone)
using OpenAI CLIP text-image embeddings.
"""

import cv2
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class ZeroShotClassifier:
    """
    Zero-shot classifier leveraging OpenAI CLIP (openai/clip-vit-base-patch32)
    to refine coarse YOLOv8 predictions for un-modeled species, birds, and drones.
    """

    def __init__(self, model_name="openai/clip-vit-base-patch32", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[ZeroShotClassifier] Initializing {model_name} on {self.device}...")

        try:
            self.model = CLIPModel.from_pretrained(model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.model.eval()
            self.is_ready = True
            print(f"[ZeroShotClassifier] Successfully loaded {model_name}.")
        except Exception as err:
            print(f"[ZeroShotClassifier Error] Failed to load {model_name}: {err}")
            self.is_ready = False

        self.default_animal_labels = [
            "panda",
            "giant panda",
            "lion",
            "tiger",
            "leopard",
            "cheetah",
            "cat",
            "dog",
            "wolf",
            "fox",
            "deer",
            "elephant",
            "bear",
            "monkey",
            "gorilla",
            "chimpanzee",
            "zebra",
            "giraffe",
            "hippopotamus",
            "rhinoceros",
            "kangaroo",
            "koala",
            "penguin",
            "horse",
            "cow",
            "sheep",
        ]

        self.default_bird_drone_labels = [
            "drone",
            "quadcopter",
            "eagle",
            "hawk",
            "parrot",
            "crow",
            "sparrow",
            "bird",
            "airplane",
            "kite",
        ]

        self.default_all_labels = list(
            set(self.default_animal_labels + self.default_bird_drone_labels)
        )

    def classify_crop(self, bgr_crop, candidate_labels=None):
        """
        Classifies a bounding box crop against candidate text labels.
        Returns: (best_label_str, confidence_float)
        """
        if not self.is_ready or bgr_crop is None or bgr_crop.size == 0:
            return None, 0.0

        h, w = bgr_crop.shape[:2]
        if h < 8 or w < 8:
            return None, 0.0

        if candidate_labels is None:
            candidate_labels = self.default_all_labels

        try:
            rgb_crop = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_crop)

            text_prompts = [f"a photo of a {label}" for label in candidate_labels]

            inputs = self.processor(
                text=text_prompts, images=pil_img, return_tensors="pt", padding=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=-1)[0]

            best_idx = probs.argmax().item()
            best_label = candidate_labels[best_idx]
            best_prob = float(probs[best_idx].item())

            return best_label, best_prob

        except Exception as err:
            print(f"[ZeroShotClassifier Error] Inference failed: {err}")
            return None, 0.0
