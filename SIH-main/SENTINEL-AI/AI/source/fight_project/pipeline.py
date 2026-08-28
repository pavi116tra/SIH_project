import os
import math
import collections
from datetime import datetime
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO

from model import X3DFightClassifier
from alert_system import AlertSystem
from weapon_detector import WeaponAndThreatDetector
from audio_flash_analyzer import detect_visual_muzzle_flash_spikes

class ResNetBiLSTMAttention(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        import torchvision.models as models
        try:
            resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        except Exception:
            resnet = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.bilstm = nn.LSTM(512, 256, num_layers=2, batch_first=True, bidirectional=True)
        self.attn = nn.Sequential(nn.Linear(512, 128), nn.Tanh(), nn.Linear(128, 1))
        self.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, num_classes))

    def forward(self, x):
        B, T, C, H, W = x.size()
        x_flat = x.view(B * T, C, H, W)
        feats = torch.flatten(self.backbone(x_flat), 1)
        seq_feats = feats.view(B, T, 512)
        lstm_out, _ = self.bilstm(seq_feats)
        weights = F.softmax(self.attn(lstm_out), dim=1)
        context = torch.sum(lstm_out * weights, dim=1)
        return self.fc(context)

def compute_optical_flow_magnitude(frames_bgr):
    if len(frames_bgr) < 2:
        return 0.0
    mags = []
    prev_gray = cv2.cvtColor(frames_bgr[0], cv2.COLOR_BGR2GRAY)
    for f in frames_bgr[1:]:
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mags.append(np.mean(mag))
        prev_gray = gray
    avg_mag = float(np.mean(mags)) if mags else 0.0
    return min(1.0, avg_mag / 5.0)

def compute_person_proximity_score(person_boxes, frame_width, frame_height):
    if len(person_boxes) < 2:
        return 0.0
    max_proximity = 0.0
    for i in range(len(person_boxes)):
        for j in range(i + 1, len(person_boxes)):
            b1 = person_boxes[i]
            b2 = person_boxes[j]
            c1 = ((b1[0] + b1[2]) / 2.0, (b1[1] + b1[3]) / 2.0)
            c2 = ((b2[0] + b2[2]) / 2.0, (b2[1] + b2[3]) / 2.0)
            dist = math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
            max_diag = math.sqrt(frame_width**2 + frame_height**2)
            dist_score = max(0.0, 1.0 - (dist / (max_diag * 0.3)))

            x1 = max(b1[0], b2[0])
            y1 = max(b1[1], b2[1])
            x2 = min(b1[2], b2[2])
            y2 = min(b1[3], b2[3])
            overlap = max(0, x2 - x1) * max(0, y2 - y1)
            area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
            area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
            union = area1 + area2 - overlap + 1e-6
            iou = overlap / union

            pair_score = max(dist_score, min(1.0, iou * 3.0))
            if pair_score > max_proximity:
                max_proximity = pair_score
    return max_proximity

class DeepMultimodalThreatPipeline:
    """
    Comprehensive Deep Threat Analysis Pipeline:
    1. Sliding-Window Video Fight Classification (X3D-S + ResNet18-BiLSTM)
    2. Weapon & Threat Object Detection (Knives, Firearms, Sharp/Blunt Weapons)
    3. Flash Spike Anomaly Detection (Gun Muzzle Flash / Sudden Explosive Bursts)
    4. Optical Flow Motion Magnitude (Kicking, Punching, Throwing, Hitting)
    5. Person Proximity & Bounding Box Overlap Analysis
    6. Multi-Signal Fusion Threat Escalation System
    """
    def __init__(self, fight_threshold=0.60, camera_id="MAIN_CAM"):
        self.camera_id = camera_id
        self.fight_threshold = fight_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.stride = 4
        self.window_size = 16

        self.mean = torch.tensor([0.45, 0.45, 0.45], device=self.device).view(1, 3, 1, 1, 1)
        self.std = torch.tensor([0.225, 0.225, 0.225], device=self.device).view(1, 3, 1, 1, 1)

        # 1. Primary X3D-S Classifier
        print(f"Loading X3D-S Video Classifier on device: {self.device}")
        self.x3d_model = X3DFightClassifier(pretrained=True).to(self.device)
        x3d_path = "fight_detector_x3d_s.pt"
        if os.path.exists(x3d_path):
            try:
                self.x3d_model.load_state_dict(torch.load(x3d_path, map_location=self.device), strict=False)
            except Exception as e:
                print(f"Notice loading '{x3d_path}': {e}")
        self.x3d_model.eval()

        # 2. Secondary ResNet18-BiLSTM Classifier
        self.bilstm_model = ResNetBiLSTMAttention().to(self.device)
        bilstm_path = "best_fight_classifier.pt"
        if os.path.exists(bilstm_path):
            try:
                self.bilstm_model.load_state_dict(torch.load(bilstm_path, map_location=self.device), strict=False)
            except Exception as e:
                print(f"Notice loading '{bilstm_path}': {e}")
        self.bilstm_model.eval()

        # 3. YOLO Person Tracker & Weapon Detector
        self.yolo = YOLO("yolov8n.pt")
        self.weapon_detector = WeaponAndThreatDetector()

        # 4. Alert & Frame Buffers
        self.alert_system = AlertSystem()
        self.frame_buffer = collections.deque(maxlen=16)
        self.frame_counter = 0
        self.last_combined_score = 0.0
        self.last_threat_level = "NORMAL"

    def preprocess_x3d_tensor(self, frames_bgr):
        rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr[:16]]
        resized_frames = [cv2.resize(f, (224, 224)) for f in rgb_frames]
        clip = np.stack(resized_frames)
        tensor = torch.from_numpy(clip).float().to(self.device) / 255.0
        tensor = tensor.permute(3, 0, 1, 2).unsqueeze(0)
        tensor = (tensor - self.mean) / self.std
        return tensor

    def preprocess_bilstm_tensor(self, frames_bgr):
        import torchvision.transforms as T
        norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        t_list = []
        for f in frames_bgr[:16]:
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            pil_img = T.functional.to_pil_image(rgb)
            pil_resized = T.functional.resize(pil_img, (224, 224))
            t = T.functional.to_tensor(pil_resized)
            t = norm(t)
            t_list.append(t)
        return torch.stack(t_list, dim=0).unsqueeze(0).to(self.device)

    def evaluate_window_threat_fusion(self, frames_16, person_boxes, frame_w, frame_h):
        """
        Deep Multi-Signal Threat Fusion:
        1. Model Ensemble (X3D-S + ResNet18-BiLSTM)
        2. Weapon Detection (Knives, Firearms, Sharp/Blunt Weapons)
        3. Optical Flow Motion (Kicking/Punching/Hitting)
        4. Person Proximity
        5. Muzzle Flash Anomaly Detection
        """
        # 1. Model Ensemble
        x3d_in = self.preprocess_x3d_tensor(frames_16)
        bilstm_in = self.preprocess_bilstm_tensor(frames_16)

        with torch.no_grad():
            x3d_logits = self.x3d_model(x3d_in)
            x3d_prob = float(F.softmax(x3d_logits, dim=1)[0, 1].cpu().numpy())

            bilstm_logits = self.bilstm_model(bilstm_in)
            bilstm_prob = float(F.softmax(bilstm_logits, dim=1)[0, 1].cpu().numpy())

        model_ensemble_conf = (0.6 * x3d_prob) + (0.4 * bilstm_prob)

        # 2. Weapon Detection on middle frame of window
        mid_f = frames_16[len(frames_16) // 2]
        weapons = self.weapon_detector.detect_weapons(mid_f)
        has_weapon = len(weapons) > 0

        # 3. Optical Flow & Person Proximity
        flow_score = compute_optical_flow_magnitude(frames_16)
        proximity_score = compute_person_proximity_score(person_boxes, frame_w, frame_h)
        physical_signal = (0.5 * flow_score) + (0.5 * proximity_score)

        # 4. Muzzle Flash Spike Check
        flash_spikes = detect_visual_muzzle_flash_spikes(frames_16)
        has_flash_spike = len(flash_spikes) > 0

        # 5. Combined Score Computation
        combined_score = (0.7 * model_ensemble_conf) + (0.3 * physical_signal)
        if has_weapon or has_flash_spike:
            combined_score = min(1.0, combined_score + 0.35)

        # 6. Rule-Based Threat Level Classification
        if has_weapon or has_flash_spike:
            threat_level = "CRITICAL WEAPON THREAT"
            is_fight = True
        elif combined_score >= self.fight_threshold or (model_ensemble_conf >= 0.30 and proximity_score >= 0.4 and flow_score >= 0.4):
            threat_level = "PHYSICAL ALTERCATION DETECTED"
            is_fight = True
        elif 0.40 <= combined_score < self.fight_threshold or (proximity_score >= 0.35 and flow_score >= 0.35):
            threat_level = "SUSPICIOUS ALTERCATION"
            is_fight = False
        else:
            threat_level = "NORMAL"
            is_fight = False

        return {
            "combined_score": round(float(combined_score), 4),
            "model_ensemble_conf": round(model_ensemble_conf, 4),
            "flow_score": round(flow_score, 4),
            "proximity_score": round(proximity_score, 4),
            "has_weapon": has_weapon,
            "weapons": weapons,
            "has_flash_spike": has_flash_spike,
            "threat_level": threat_level,
            "is_fight": is_fight
        }

    def analyze_full_video_deep(self, all_frames, fps=25.0):
        """
        Performs Deep Frame-by-Frame & Window-by-Window Threat Analysis across full video.
        Returns full temporal threat timeline, weapon log, and frame status.
        """
        total_frames = len(all_frames)
        if total_frames < 16:
            raise ValueError(f"Video clip is too short ({total_frames} frames < 16 required frames).")

        h, w = all_frames[0].shape[:2]
        window_evaluations = []
        threat_spans = []

        active_threat_span = None
        frame_threat_status = ["NORMAL"] * total_frames
        frame_weapons_log = [[] for _ in range(total_frames)]

        for start_idx in range(0, total_frames - 15, self.stride):
            end_idx = start_idx + 16
            window_frames = all_frames[start_idx:end_idx]

            mid_f = window_frames[len(window_frames) // 2]
            yolo_res = self.yolo(mid_f, classes=[0], verbose=False)
            p_boxes = []
            if yolo_res and len(yolo_res) > 0 and yolo_res[0].boxes is not None:
                for b in yolo_res[0].boxes:
                    p_boxes.append(list(map(int, b.xyxy[0].cpu().numpy())))

            eval_res = self.evaluate_window_threat_fusion(window_frames, p_boxes, w, h)
            start_sec = round(start_idx / fps, 2)
            end_sec = round(end_idx / fps, 2)

            eval_res["start_sec"] = start_sec
            eval_res["end_sec"] = end_sec
            window_evaluations.append(eval_res)

            t_level = eval_res["threat_level"]

            # Store frame status
            for idx_f in range(start_idx, end_idx):
                frame_threat_status[idx_f] = t_level
                if eval_res["has_weapon"]:
                    frame_weapons_log[idx_f].extend(eval_res["weapons"])

            if t_level in ["CRITICAL WEAPON THREAT", "PHYSICAL ALTERCATION DETECTED"]:
                if active_threat_span is None:
                    active_threat_span = {
                        "start_sec": start_sec,
                        "end_sec": end_sec,
                        "threat_level": t_level,
                        "max_score": eval_res["combined_score"],
                        "weapons": eval_res["weapons"]
                    }
                else:
                    active_threat_span["end_sec"] = end_sec
                    active_threat_span["max_score"] = max(active_threat_span["max_score"], eval_res["combined_score"])
                    if eval_res["has_weapon"]:
                        active_threat_span["weapons"].extend(eval_res["weapons"])
            else:
                if active_threat_span is not None:
                    threat_spans.append(active_threat_span)
                    active_threat_span = None

        if active_threat_span is not None:
            threat_spans.append(active_threat_span)

        max_score = max([w["combined_score"] for w in window_evaluations]) if window_evaluations else 0.0
        all_weapons_found = [w for win in window_evaluations for w in win["weapons"]]
        overall_threat = "NORMAL"

        if any(w["threat_level"] == "CRITICAL WEAPON THREAT" for w in window_evaluations):
            overall_threat = "CRITICAL WEAPON THREAT"
        elif any(w["threat_level"] == "PHYSICAL ALTERCATION DETECTED" for w in window_evaluations):
            overall_threat = "PHYSICAL ALTERCATION DETECTED"
        elif any(w["threat_level"] == "SUSPICIOUS ALTERCATION" for w in window_evaluations):
            overall_threat = "SUSPICIOUS ALTERCATION"

        return {
            "overall_threat_level": overall_threat,
            "max_confidence": max_score,
            "threat_spans": threat_spans,
            "window_evaluations": window_evaluations,
            "frame_threat_status": frame_threat_status,
            "frame_weapons_log": frame_weapons_log,
            "all_weapons_found": all_weapons_found
        }

    def process_frame(self, frame_bgr):
        """
        Live RTSP / Webcam Frame Processor with Deep Threat HUD Rendering.
        """
        if frame_bgr is None:
            raise ValueError("Null frame received.")

        h, w = frame_bgr.shape[:2]
        self.frame_buffer.append(frame_bgr.copy())
        self.frame_counter += 1

        yolo_res = self.yolo(frame_bgr, classes=[0], verbose=False)
        person_boxes = []
        if yolo_res and len(yolo_res) > 0 and yolo_res[0].boxes is not None:
            for b in yolo_res[0].boxes:
                person_boxes.append(list(map(int, b.xyxy[0].cpu().numpy())))

        eval_res = None
        if len(self.frame_buffer) == 16 and (self.frame_counter % self.stride == 0):
            eval_res = self.evaluate_window_threat_fusion(list(self.frame_buffer), person_boxes, w, h)
            self.last_combined_score = eval_res["combined_score"]
            self.last_threat_level = eval_res["threat_level"]

        annotated_frame = frame_bgr.copy()
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        alert_info = None

        if self.last_threat_level == "CRITICAL WEAPON THREAT":
            # Purple / Red Alert HUD
            cv2.rectangle(annotated_frame, (0, 0), (w, 45), (128, 0, 128), -1)
            cv2.putText(annotated_frame, f"CRITICAL THREAT: WEAPON / FLASH DETECTED | Conf: {self.last_combined_score*100:.1f}%",
                        (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            for (x1, y1, x2, y2) in person_boxes:
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

        elif self.last_threat_level == "PHYSICAL ALTERCATION DETECTED":
            # Red Alert HUD
            cv2.rectangle(annotated_frame, (0, 0), (w, 45), (0, 0, 220), -1)
            cv2.putText(annotated_frame, f"!!! PHYSICAL ALTERCATION DETECTED !!! Conf: {self.last_combined_score*100:.1f}%",
                        (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            for (x1, y1, x2, y2) in person_boxes:
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

        elif self.last_threat_level == "SUSPICIOUS ALTERCATION":
            # Orange HUD
            cv2.rectangle(annotated_frame, (0, 0), (w, 35), (0, 165, 255), -1)
            cv2.putText(annotated_frame, f"SUSPICIOUS MOTION | Score: {self.last_combined_score*100:.1f}%",
                        (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            for (x1, y1, x2, y2) in person_boxes:
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 165, 255), 2)

        else:
            # Normal Green HUD
            cv2.rectangle(annotated_frame, (0, 0), (w, 35), (40, 40, 40), -1)
            cv2.putText(annotated_frame, f"Status: NORMAL | Threat Score: {self.last_combined_score*100:.1f}%",
                        (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            for (x1, y1, x2, y2) in person_boxes:
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return annotated_frame, (self.last_threat_level in ["CRITICAL WEAPON THREAT", "PHYSICAL ALTERCATION DETECTED"]), self.last_combined_score, alert_info

# Backward compatibility aliases
EnsembleFightDetectionPipeline = DeepMultimodalThreatPipeline
X3DFightDetectionPipeline = DeepMultimodalThreatPipeline
FightDetectionPipeline = DeepMultimodalThreatPipeline

if __name__ == "__main__":
    p = DeepMultimodalThreatPipeline()
    print("DeepMultimodalThreatPipeline initialized successfully.")
