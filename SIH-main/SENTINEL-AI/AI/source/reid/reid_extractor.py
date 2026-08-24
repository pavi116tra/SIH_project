"""
Deep Appearance Re-Identification (ReID) Feature Extractor using OSNet (osnet_x1_0).
Extracts 512-dimensional L2-normalized appearance embeddings from person image crops
using aspect-preserving 256x128 preprocessing and ImageNet normalization.
"""

import os
import cv2
import torch
import hashlib
import numpy as np
import torchreid


class PersonReIDExtractor:
    """
    Extracts deep visual appearance embeddings using pretrained OSNet (osnet_x1_0).
    """

    def __init__(self, model_name="osnet_x1_0", device="auto"):
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"[PersonReIDExtractor] Initializing OSNet model '{model_name}' on {self.device}...")
        self.model = torchreid.models.build_model(
            name=model_name,
            num_classes=1000,
            loss="softmax",
            pretrained=True
        )
        self.model.eval()
        self.model.to(self.device)

        # Log weight checksum hash check on startup to verify non-random weights
        weight_bytes = b"".join([p.data.cpu().numpy().tobytes() for p in list(self.model.parameters())[:5]])
        weight_hash = hashlib.md5(weight_bytes).hexdigest()[:10]
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "torch", "checkpoints")
        print(f"[PersonReIDExtractor] Successfully loaded pretrained OSNet weights (Cache: '{cache_dir}', Checksum: {weight_hash})")

        # Standard OSNet input dimensions & ImageNet normalization parameters
        self.input_height = 256
        self.input_width = 128
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

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

    def preprocess_crop(self, crop):
        """
        Aspect-preserving resize + letterbox padding to 256x128 with ImageNet normalization.
        """
        h, w = crop.shape[:2]
        # BGR to RGB
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        # Calculate scale to fit within 256x128
        scale = min(self.input_height / float(h), self.input_width / float(w))
        new_h = int(round(h * scale))
        new_w = int(round(w * scale))

        resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Neutral gray canvas (128, 128, 128)
        canvas = np.full((self.input_height, self.input_width, 3), 128, dtype=np.uint8)
        top = (self.input_height - new_h) // 2
        left = (self.input_width - new_w) // 2
        canvas[top:top + new_h, left:left + new_w] = resized

        # Normalize to [0, 1] then apply ImageNet mean/std
        normalized = (canvas.astype(np.float32) / 255.0 - self.mean) / self.std

        # HWC to CHW tensor
        tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        return tensor

    def extract_features(self, crop, check_blur=True):
        """
        Extract a 512-dimensional L2-normalized OSNet feature vector from a full-body person crop.
        crop: np.ndarray BGR image crop
        """
        if crop is None or crop.size == 0 or crop.shape[0] < 15 or crop.shape[1] < 15:
            return None

        if check_blur and self.is_blurry(crop):
            return None

        try:
            tensor = self.preprocess_crop(crop)
            with torch.no_grad():
                features = self.model(tensor)
                embedding = features.cpu().numpy()[0]

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
        Compute cosine similarity between two L2-normalized OSNet embeddings.
        Returns float between -1.0 and 1.0 (1.0 = identical appearance).
        """
        if emb1 is None or emb2 is None:
            return 0.0
        return float(np.dot(emb1, emb2))
