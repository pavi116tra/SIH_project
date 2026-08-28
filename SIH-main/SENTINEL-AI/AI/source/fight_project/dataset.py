import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class X3DVideoDataset(Dataset):
    """
    PyTorch Dataset for X3D-S video fight classification.
    Inputs: [3, 16, 224, 224] (RGB, float 0..1)
    Output: label ID (0 = no_fight, 1 = fight)
    """
    def __init__(self, sample_list, num_frames=16, is_train=True):
        self.samples = sample_list
        self.num_frames = num_frames
        self.is_train = is_train
        self.label_map = {"NonFight": 0, "no_fight": 0, "Fight": 1, "fight": 1}

    def __len__(self):
        return len(self.samples)

    def _read_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []

        if not cap.isOpened():
            # Return dummy zero frames if unreadable
            return [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(self.num_frames)]

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, max(0, total_frames - 1), self.num_frames, dtype=int) if total_frames > 0 else range(self.num_frames)

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_resized = cv2.resize(rgb, (224, 224))
                frames.append(rgb_resized)
            else:
                frames.append(np.zeros((224, 224, 3), dtype=np.uint8))

        cap.release()

        while len(frames) < self.num_frames:
            frames.append(frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8))

        return frames[:self.num_frames]

    def __getitem__(self, idx):
        sample = self.samples[idx]
        video_path = sample["path"]
        raw_label = sample["label"]
        label_id = self.label_map.get(raw_label, 0)

        rgb_frames = self._read_frames(video_path)
        
        # Shape: (16, 224, 224, 3)
        clip = np.stack(rgb_frames)
        
        # Convert to float tensor and normalize to 0..1
        tensor = torch.from_numpy(clip).float() / 255.0
        
        # Permute from (T, H, W, C) to (C, T, H, W) -> (3, 16, 224, 224)
        tensor = tensor.permute(3, 0, 1, 2)

        return tensor, torch.tensor(label_id, dtype=torch.long)

VideoDataset = X3DVideoDataset
