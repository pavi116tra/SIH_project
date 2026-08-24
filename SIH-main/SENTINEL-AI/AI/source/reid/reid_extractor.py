"""
Deep Appearance Re-Identification (ReID) Feature Extractor.
Extracts 512-dimensional L2-normalized appearance embeddings from person image crops
and computes cosine similarity for cross-camera and temporal identity association.
"""

import cv2
import torch
import numpy as np
from facenet_pytorch import InceptionResnetV1


class PersonReIDExtractor:
    """
    Extracts deep visual appearance embeddings for person re-identification.
    """

    def __init__(self, device="auto"):
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"[PersonReIDExtractor] Initializing ReID backbone on {self.device}...")
        self.model = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)

    @staticmethod
    def is_blurry(crop, threshold=15.0):
        """Variance of Laplacian blur check to filter out extreme motion blur crops."""
        if crop is None or crop.size == 0:
            return True
        try:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            val = cv2.Laplacian(gray, cv2.CV_64F).var()
            return val < threshold
        except Exception:
            return False

    def extract_features(self, crop, check_blur=True):
        """
        Extract a 512-dimensional L2-normalized feature vector from a person crop.
        crop: np.ndarray BGR image crop
        """
        if crop is None or crop.size == 0 or crop.shape[0] < 15 or crop.shape[1] < 15:
            return None

        if check_blur and self.is_blurry(crop):
            return None

        try:
            h, w = crop.shape[:2]
            # Focus on upper-body / head region (top 65% of person bounding box)
            upper_crop = crop[:max(int(h * 0.65), 15), :]

            rgb_upper = cv2.cvtColor(upper_crop, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb_upper, (160, 160))

            # Normalize to [-1, 1] range
            tensor = (
                torch.tensor(resized, dtype=torch.float32).permute(2, 0, 1) - 127.5
            ) / 128.0
            batch = tensor.unsqueeze(0).to(self.device)

            with torch.no_grad():
                embedding = self.model(batch).cpu().numpy()[0]

            # L2 normalization
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            return embedding
        except Exception as e:
            print(f"[PersonReIDExtractor] Extraction error: {e}")
            return None

    @staticmethod
    def compute_similarity(emb1, emb2):
        """
        Compute cosine similarity between two L2-normalized embeddings.
        Returns float between -1.0 and 1.0 (1.0 = identical appearance).
        """
        if emb1 is None or emb2 is None:
            return 0.0
        return float(np.dot(emb1, emb2))
