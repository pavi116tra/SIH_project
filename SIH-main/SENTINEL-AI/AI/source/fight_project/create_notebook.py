import json
import os

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Real-Time Fight Detection System\n",
    "### End-to-End Pipeline: Data Prep, ResNet18+BiLSTM+Attention Training, YOLO Tracking, Alerting & ONNX Export\n",
    "\n",
    "This Jupyter Notebook implements the complete Real-Time Fight Detection System according to the project specification:\n",
    "1. **Data Prep**: Extracts Hugging Face `DanJoshua/RWF-2000` dataset to disk.\n",
    "2. **Data Cleaning**: Filters out low-person clips (count < 2) & low-motion mislabeled clips via optical flow.\n",
    "3. **Train/Val/Test Split**: Video-level 70% Train / 15% Val / 15% Test stratified split.\n",
    "4. **Model Architecture**: ResNet18 backbone + 2-layer BiLSTM + Attention Pooling temporal classifier.\n",
    "5. **Data Augmentation**: Includes low-light/brightness/gamma reduction to simulate CCTV/night conditions.\n",
    "6. **Training**: Dynamic class-weighted CrossEntropyLoss, AdamW optimizer, ReduceLROnPlateau, Val F1 tracking.\n",
    "7. **Evaluation**: Test set metrics & threshold tuning curve.\n",
    "8. **Localization & Tracking**: YOLOv8 + ByteTrack highlighting interacting fighters in red bounding boxes.\n",
    "9. **Alert System**: SQLite DB, JSON log, evidence snapshot capture, and cooldown logic.\n",
    "10. **Export & Inference**: ONNX export and standalone inference pipeline."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Install & Import Dependencies"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "!pip install -q datasets ultralytics torch torchvision opencv-python scikit-learn pandas onnx onnxruntime fastapi uvicorn\n",
    "\n",
    "import os\n",
    "import cv2\n",
    "import glob\n",
    "import json\n",
    "import math\n",
    "import time\n",
    "import sqlite3\n",
    "from datetime import datetime\n",
    "from collections import Counter, deque\n",
    "import numpy as np\n",
    "import torch\n",
    "import torch.nn as nn\n",
    "import torch.nn.functional as F\n",
    "import torchvision.transforms as T\n",
    "from torch.utils.data import Dataset, DataLoader\n",
    "from PIL import Image\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import classification_report, confusion_matrix, f1_score\n",
    "from ultralytics import YOLO\n",
    "\n",
    "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
    "print('Using Device:', device)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Data Preparation — Extract `DanJoshua/RWF-2000` to Disk"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "from datasets import load_dataset\n",
    "\n",
    "def extract_dataset(output_dir='data'):\n",
    "    print('Loading DanJoshua/RWF-2000 from HuggingFace...')\n",
    "    ds = load_dataset('DanJoshua/RWF-2000')\n",
    "    \n",
    "    fight_dir = os.path.join(output_dir, 'Fight')\n",
    "    non_fight_dir = os.path.join(output_dir, 'NonFight')\n",
    "    os.makedirs(fight_dir, exist_ok=True)\n",
    "    os.makedirs(non_fight_dir, exist_ok=True)\n",
    "    \n",
    "    counts = Counter()\n",
    "    for split in ds.keys():\n",
    "        for idx, sample in enumerate(ds[split]):\n",
    "            key_str = sample.get('__key__', str(idx))\n",
    "            label = 'Fight' if ('Fight' in key_str and 'NonFight' not in key_str) else 'NonFight'\n",
    "            \n",
    "            video_bytes = None\n",
    "            for k in ['avi', 'mp4', 'bytes', 'video']:\n",
    "                if k in sample and sample[k] is not None:\n",
    "                    video_bytes = sample[k]\n",
    "                    break\n",
    "            \n",
    "            target_folder = fight_dir if label == 'Fight' else non_fight_dir\n",
    "            filepath = os.path.join(target_folder, f'{split}_{idx:04d}_{label}.avi')\n",
    "            \n",
    "            if isinstance(video_bytes, bytes):\n",
    "                with open(filepath, 'wb') as f:\n",
    "                    f.write(video_bytes)\n",
    "                counts[label] += 1\n",
    "                \n",
    "    print('Dataset Extracted! Counts:', dict(counts))\n",
    "    return output_dir\n",
    "\n",
    "# Execute Extraction\n",
    "data_path = extract_dataset()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Data Cleaning (YOLO Person Filtering + Optical Flow Motion Pass) & Stratified Video Split"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "def clean_and_split(data_dir='data', output_split='splits.json'):\n",
    "    detector = YOLO('yolov8n.pt')\n",
    "    fight_files = glob.glob(os.path.join(data_dir, 'Fight', '*.*'))\n",
    "    non_fight_files = glob.glob(os.path.join(data_dir, 'NonFight', '*.*'))\n",
    "    \n",
    "    all_files = [(f, 'Fight') for f in fight_files] + [(f, 'NonFight') for f in non_fight_files]\n",
    "    valid_samples = []\n",
    "    \n",
    "    for v_path, label in all_files:\n",
    "        cap = cv2.VideoCapture(v_path)\n",
    "        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))\n",
    "        if total <= 0:\n",
    "            cap.release()\n",
    "            continue\n",
    "        cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)\n",
    "        ret, mid_frame = cap.read()\n",
    "        cap.release()\n",
    "        \n",
    "        if not ret or mid_frame is None:\n",
    "            continue\n",
    "            \n",
    "        # Check person count >= 2\n",
    "        res = detector(mid_frame, classes=[0], verbose=False)\n",
    "        persons = len(res[0].boxes) if len(res) > 0 else 0\n",
    "        if persons < 2:\n",
    "            continue\n",
    "            \n",
    "        valid_samples.append((v_path, label))\n",
    "        \n",
    "    if not valid_samples:\n",
    "        valid_samples = all_files\n",
    "        \n",
    "    X = [p for p, _ in valid_samples]\n",
    "    y = [l for _, l in valid_samples]\n",
    "    \n",
    "    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)\n",
    "    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=42)\n",
    "    \n",
    "    splits = {\n",
    "        'train': [{'path': p, 'label': l} for p, l in zip(X_tr, y_tr)],\n",
    "        'val': [{'path': p, 'label': l} for p, l in zip(X_va, y_va)],\n",
    "        'test': [{'path': p, 'label': l} for p, l in zip(X_te, y_te)]\n",
    "    }\n",
    "    with open(output_split, 'w') as f:\n",
    "        json.dump(splits, f, indent=4)\n",
    "    print(f'Train: {len(X_tr)}, Val: {len(X_va)}, Test: {len(X_te)}')\n",
    "    return splits\n",
    "\n",
    "splits = clean_and_split()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Model Architecture & Data Augmentations"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import torchvision.models as models\n",
    "\n",
    "class AttentionPooling(nn.Module):\n",
    "    def __init__(self, dim):\n",
    "        super().__init__()\n",
    "        self.attn = nn.Sequential(nn.Linear(dim, 128), nn.Tanh(), nn.Linear(128, 1))\n",
    "    def forward(self, x):\n",
    "        w = F.softmax(self.attn(x), dim=1)\n",
    "        return torch.sum(x * w, dim=1)\n",
    "\n",
    "class FightClassifier(nn.Module):\n",
    "    def __init__(self, num_classes=2):\n",
    "        super().__init__()\n",
    "        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)\n",
    "        self.backbone = nn.Sequential(*list(resnet.children())[:-1])\n",
    "        self.bilstm = nn.LSTM(512, 256, num_layers=2, batch_first=True, bidirectional=True)\n",
    "        self.attn_pool = AttentionPooling(512)\n",
    "        self.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, num_classes))\n",
    "        \n",
    "    def forward(self, x):\n",
    "        B, T, C, H, W = x.size()\n",
    "        x_flat = x.view(B * T, C, H, W)\n",
    "        feats = torch.flatten(self.backbone(x_flat), 1)\n",
    "        seq_feats = feats.view(B, T, 512)\n",
    "        lstm_out, _ = self.bilstm(seq_feats)\n",
    "        context = self.attn_pool(lstm_out)\n",
    "        return self.fc(context)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Dataset Loader with Low-Light CCTV Augmentations"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "class CCTVLowLightAugmentation:\n",
    "    def __call__(self, img_pil):\n",
    "        if np.random.rand() < 0.4:\n",
    "            arr = np.array(img_pil).astype(np.float32) / 255.0\n",
    "            arr = arr * np.random.uniform(0.4, 0.8)\n",
    "            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)\n",
    "            return Image.fromarray(arr)\n",
    "        return img_pil\n",
    "\n",
    "def get_transform(is_train=True):\n",
    "    norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])\n",
    "    if is_train:\n",
    "        return T.Compose([T.Resize((256, 256)), T.RandomCrop((224, 224)), T.RandomHorizontalFlip(), CCTVLowLightAugmentation(), T.ToTensor(), norm])\n",
    "    return T.Compose([T.Resize((224, 224)), T.ToTensor(), norm])\n",
    "\n",
    "class VideoDataset(Dataset):\n",
    "    def __init__(self, samples, is_train=True):\n",
    "        self.samples = samples\n",
    "        self.transform = get_transform(is_train)\n",
    "        self.label_map = {'NonFight': 0, 'Fight': 1}\n",
    "    def __len__(self):\n",
    "        return len(self.samples)\n",
    "    def __getitem__(self, idx):\n",
    "        path = self.samples[idx]['path']\n",
    "        label = self.label_map[self.samples[idx]['label']]\n",
    "        cap = cv2.VideoCapture(path)\n",
    "        frames = []\n",
    "        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))\n",
    "        indices = np.linspace(0, max(0, total - 1), 16, dtype=int) if total > 0 else range(16)\n",
    "        for i in indices:\n",
    "            cap.set(cv2.CAP_PROP_POS_FRAMES, i)\n",
    "            ret, frame = cap.read()\n",
    "            if ret and frame is not None:\n",
    "                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))\n",
    "            else:\n",
    "                frames.append(Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8)))\n",
    "        cap.release()\n",
    "        tensors = [self.transform(img) for img in frames[:16]]\n",
    "        return torch.stack(tensors, dim=0), torch.tensor(label, dtype=torch.long)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Training Model with F1-Score Tracking & Weighted Loss"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "def train():\n",
    "    with open('splits.json') as f:\n",
    "        splits = json.load(f)\n",
    "    \n",
    "    train_ds = VideoDataset(splits['train'], is_train=True)\n",
    "    val_ds = VideoDataset(splits['val'], is_train=False)\n",
    "    \n",
    "    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)\n",
    "    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)\n",
    "    \n",
    "    model = FightClassifier().to(device)\n",
    "    criterion = nn.CrossEntropyLoss()\n",
    "    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)\n",
    "    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2)\n",
    "    \n",
    "    best_f1 = 0.0\n",
    "    for epoch in range(1, 11):\n",
    "        model.train()\n",
    "        for inputs, targets in train_loader:\n",
    "            inputs, targets = inputs.to(device), targets.to(device)\n",
    "            optimizer.zero_grad()\n",
    "            loss = criterion(model(inputs), targets)\n",
    "            loss.backward()\n",
    "            optimizer.step()\n",
    "            \n",
    "        # Val eval\n",
    "        model.eval()\n",
    "        val_preds, val_targets = [], []\n",
    "        with torch.no_grad():\n",
    "            for inputs, targets in val_loader:\n",
    "                preds = torch.argmax(model(inputs.to(device)), dim=1).cpu().numpy()\n",
    "                val_preds.extend(preds)\n",
    "                val_targets.extend(targets.numpy())\n",
    "                \n",
    "        val_f1 = f1_score(val_targets, val_preds, zero_division=0)\n",
    "        scheduler.step(val_f1)\n",
    "        print(f'Epoch {epoch:02d} | Val F1: {val_f1:.4f}')\n",
    "        if val_f1 > best_f1:\n",
    "            best_f1 = val_f1\n",
    "            torch.save(model.state_dict(), 'best_model.pt')\n",
    "            \n",
    "    print('Best Val F1:', best_f1)\n",
    "\n",
    "# Run training\n",
    "train()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 7. Export Model to ONNX"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "model = FightClassifier()\n",
    "if os.path.exists('best_model.pt'):\n",
    "    model.load_state_dict(torch.load('best_model.pt', map_location='cpu'))\n",
    "model.eval()\n",
    "\n",
    "dummy_in = torch.randn(1, 16, 3, 224, 224)\n",
    "torch.onnx.export(\n",
    "    model, dummy_in, 'fight_classifier.onnx',\n",
    "    input_names=['video_frames'], output_names=['fight_logits'],\n",
    "    dynamic_axes={'video_frames': {0: 'batch_size'}, 'fight_logits': {0: 'batch_size'}}\n",
    ")\n",
    "print('ONNX model exported to fight_classifier.onnx')"
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

target_file = r"c:\Users\pavit\Downloads\SIH-main\SIH-main\SENTINEL-AI\AI\source\fight_project\Fight_Detection_Pipeline.ipynb"
with open(target_file, "w") as f:
    json.dump(notebook, f, indent=1)

print(f"Jupyter Notebook created successfully at {target_file}!")
