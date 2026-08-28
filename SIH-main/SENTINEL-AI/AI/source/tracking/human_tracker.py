"""
High-Accuracy Human Detection, Tracking, and Analytics Pipeline Manager.
Coordinates YOLO object detection, ByteTrack multi-object association,
and session-wide unique counting.
"""

import time
import torch
import yaml
import os
import threading
import numpy as np
from ultralytics import YOLO
from tracking.tracker import ByteTracker


class HumanTrackerEngine:
    """
    High-Accuracy Real-Time Human Detection & Multi-Object Tracking Engine.
    Filters target 'person' class from non-human objects, assigns persistent IDs,
    maintains unique observation stats, and collects real-time analytics.
    """

    def __init__(self, config_path=None):
        self.lock = threading.Lock()
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        if config_path is None:
            config_path = os.path.join(self.base_dir, "config", "config.yaml")

        self.config = self._load_config(config_path)

        # Device configuration
        device_cfg = self.config.get("model", {}).get("device", "auto")
        if device_cfg == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device_cfg

        # Load YOLO detector
        detector_path = self.config.get("model", {}).get("detector_path", "yolov8n.pt")
        if not os.path.isabs(detector_path):
            abs_det_path = os.path.join(self.base_dir, detector_path)
            if os.path.exists(abs_det_path):
                detector_path = abs_det_path

        print(f"[HumanTracker] Loading detector: {detector_path} on {self.device}")
        self.detector = YOLO(detector_path)
        self.detector.to(self.device)

        # Configuration parameters
        self.conf_thresh = self.config.get("model", {}).get("confidence_threshold", 0.40)
        self.iou_thresh = self.config.get("model", {}).get("iou_threshold", 0.50)
        self.img_size = self.config.get("model", {}).get("img_size", 640)
        self.target_class_id = self.config.get("classes", {}).get("target_class_id", 0)

        # Initialize independent per-camera ByteTracker map
        self.trackers = {}

        # Initialize Global Identity Manager (Level 2 Identity: P001, P002...)
        from tracking.global_id_manager import GlobalIDManager
        global_cfg = self.config.get("global_id", {})
        self.global_id_manager = GlobalIDManager(
            retention_minutes=global_cfg.get("retention_minutes", 60),
            accept_threshold=global_cfg.get("accept_threshold", 0.88),
            reject_threshold=global_cfg.get("reject_threshold", 0.55),
            merge_threshold=global_cfg.get("merge_threshold", 0.88),
            max_prototypes=global_cfg.get("max_prototypes", 8),
            cooldown_seconds=global_cfg.get("cooldown_seconds", 10.0),
            max_pending_wait_seconds=global_cfg.get("max_pending_wait_seconds", 5.0),
            device=self.device,
        )

        # Initialize Zero-Shot CLIP Classifier for Fine-Grained Species & Drone Detection
        try:
            from model.zero_shot_classifier import ZeroShotClassifier
            self.zero_shot_classifier = ZeroShotClassifier(device=self.device)
        except Exception as z_err:
            print(f"[HumanTracker] ZeroShotClassifier disabled: {z_err}")
            self.zero_shot_classifier = None

        # Analytics tracking states
        self.unique_human_ids = set()
        self.previous_active_tracks = {}  # {camera_id: set_of_track_ids}
        self.frame_count = 0
        self.start_time = time.time()
        self.last_fps = 0.0
        self.last_latency_ms = 0.0

    def _get_tracker(self, camera_id):
        """Retrieve or create an independent ByteTracker instance for the camera (Thread-Safe)."""
        with self.lock:
            if camera_id not in self.trackers:
                tracker_cfg = self.config.get("tracker", {})
                self.trackers[camera_id] = ByteTracker(
                    track_high_thresh=tracker_cfg.get("track_high_thresh", 0.50),
                    track_low_thresh=tracker_cfg.get("track_low_thresh", 0.15),
                    new_track_thresh=tracker_cfg.get("new_track_thresh", 0.60),
                    match_thresh=tracker_cfg.get("match_thresh", 0.80),
                    track_buffer=tracker_cfg.get("track_buffer", 30),
                    confirm_frames=tracker_cfg.get("confirm_frames", 1),
                )
                print(f"[HumanTrackerEngine] Created independent ByteTracker for {camera_id}")
            return self.trackers[camera_id]

    def reset_camera_tracker(self, camera_id):
        """Reset ByteTracker instance and local mapping for a specific camera on loop/restart."""
        with self.lock:
            if camera_id in self.trackers:
                del self.trackers[camera_id]
            self.previous_active_tracks[camera_id] = set()
            self.global_id_manager.reset_camera_tracks(camera_id)
            print(f"[HumanTrackerEngine] Reset tracker for {camera_id}.")

    def _load_config(self, config_path):
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        return {}

    def process_frame(self, frame, camera_id="CAM01", source_type="file", suspect_map=None, faces=None):
        """
        Process a single image frame through detection, class filtering, camera-specific tracking,
        and Two-Level Global ID association (P001, P002...).
        """
        t0 = time.time()
        self.frame_count += 1

        # 1. Run YOLO object detection with thread lock on PyTorch inference (conf=0.18 to capture animals on phone screens & blurry crops)
        with self.lock:
            results = self.detector(
                frame,
                conf=0.18,
                iou=self.iou_thresh,
                imgsz=self.img_size,
                device=self.device,
                verbose=False,
            )

        # COCO class IDs for living organisms & candidate drone classes (Airplane, Bird, Cat, Dog, Horse, Sheep, Cow, Elephant, Bear, Zebra, Giraffe, Kite)
        living_organism_ids = set(self.config.get("classes", {}).get("living_organism_ids", [0, 4, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 33]))

        human_detections = []
        non_human_objects = []

        if results:
            first_res = results[0] if isinstance(results, (list, tuple)) else next(iter(results), None)
            boxes_obj = getattr(first_res, "boxes", None) if first_res is not None else None

            if boxes_obj is not None:
                for box in boxes_obj:
                    # Safe extraction for Ultralytics YOLO Box objects
                    cls_val = getattr(box.cls, "item", lambda: box.cls)()
                    cls_id = int(cls_val)
                    score_val = getattr(box.conf, "item", lambda: box.conf)()
                    score = float(score_val)

                    xyxy_attr = getattr(box, "xyxy", None)
                    if xyxy_attr is not None:
                        if hasattr(xyxy_attr, "ndim") and xyxy_attr.ndim > 1:
                            xyxy_vec = xyxy_attr[0]
                        elif isinstance(xyxy_attr, (list, tuple)) and len(xyxy_attr) > 0 and isinstance(xyxy_attr[0], (list, tuple)):
                            xyxy_vec = xyxy_attr[0]
                        else:
                            xyxy_vec = xyxy_attr

                        if hasattr(xyxy_vec, "cpu"):
                            xyxy = xyxy_vec.cpu().tolist()
                        elif hasattr(xyxy_vec, "tolist"):
                            xyxy = xyxy_vec.tolist()
                        else:
                            xyxy = list(xyxy_vec)
                    else:
                        continue

                    # Ignore inanimate/non-living objects (toothbrushes, pens, cell phones, chairs, etc.)
                    if cls_id not in living_organism_ids:
                        continue

                    det_names = getattr(self.detector, "names", {})
                    class_name = str(det_names.get(cls_id, f"Class_{cls_id}")).title()

                    if cls_id == self.target_class_id:
                        if score >= 0.18:
                            human_detections.append({
                                "bbox": xyxy,
                                "score": score,
                                "class_id": cls_id,
                                "class_name": "Person"
                            })
                    else:
                        # Non-human Organisms or Drones
                        c_name_lower = class_name.lower()
                        display_name = "Drone" if ("drone" in c_name_lower or "quadcopter" in c_name_lower) else "Animal"
                        non_human_objects.append({
                            "bbox": xyxy,
                            "score": score,
                            "class_id": cls_id,
                            "class_name": display_name
                        })

        # 1a. Fallback: If FaceNet detected faces on this frame, synthesize human body bounding boxes if YOLO missed them!
        if faces:
            h_f, w_f = frame.shape[:2]
            for face_info in faces:
                if isinstance(face_info, dict) and "bbox" in face_info:
                    fx, fy, fw, fh = face_info["bbox"]
                    fcx, fcy = fx + fw // 2, fy + fh // 2
                    px1 = max(0, fx - int(fw * 0.3))
                    py1 = max(0, fy - int(fh * 0.2))
                    px2 = min(w_f, fx + fw + int(fw * 0.3))
                    py2 = min(h_f, fy + fh + int(fh * 2.2))

                    already_covered = False
                    for h_det in human_detections:
                        hx1, hy1, hx2, hy2 = map(int, h_det["bbox"])
                        if hx1 <= fcx <= hx2 and hy1 <= fcy <= hy2:
                            already_covered = True
                            break

                    if not already_covered:
                        human_detections.append({
                            "bbox": [px1, py1, px2, py2],
                            "score": 0.88,
                            "class_id": 0,
                            "class_name": "Person"
                        })

        # 1b. Zero-Shot CLIP Refinement for Animals, Birds, and Drones
        if self.zero_shot_classifier and self.zero_shot_classifier.is_ready and non_human_objects:
            h_f, w_f = frame.shape[:2]
            for obj in non_human_objects:
                x1, y1, x2, y2 = map(int, obj["bbox"])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w_f, x2), min(h_f, y2)
                bgr_crop = frame[y1:y2, x1:x2]

                if bgr_crop.size > 0 and bgr_crop.shape[0] >= 10 and bgr_crop.shape[1] >= 10:
                    refined_label, confidence = self.zero_shot_classifier.classify_crop(bgr_crop)
                    if refined_label and confidence > 0.25:
                        r_lower = refined_label.lower()
                        obj["class_name"] = "Drone" if ("drone" in r_lower or "quadcopter" in r_lower) else "Animal"
                        obj["score"] = confidence
                        obj["is_refined"] = True

        # 2. Pass human detections to camera's independent ByteTracker (Level 1 Local Tracking)
        tracker = self._get_tracker(camera_id)
        active_human_tracks = tracker.update(human_detections)

        # 2b. Map stable, persistent Track IDs and Global IDs (NO re-indexing or renumbering)
        current_active_set = set()
        for track in active_human_tracks:
            track.camera_id = camera_id
            local_id = track.track_id
            current_active_set.add(local_id)

            suspect_name = suspect_map.get(local_id) if suspect_map else None

            # Level 2 Identity: Resolve Global Person ID (P001, P002...) via ReID / Suspect Anchoring
            x1, y1, x2, y2 = map(int, track.bbox)
            h_f, w_f = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_f, x2), min(h_f, y2)
            person_crop = frame[y1:y2, x1:x2] if (x2 > x1 and y2 > y1) else None

            global_id = self.global_id_manager.get_or_assign_global_id(
                person_crop, camera_id, local_id, source_type=source_type, suspect_name=suspect_name
            )
            track.global_id = global_id
            track.suspect_name = suspect_name

            prev_set = self.previous_active_tracks.get(camera_id, set())
            if local_id not in prev_set:
                print(f"[NEW TRACK] Camera={camera_id} TrackID={local_id} GlobalID={global_id}")

        # Detect lost tracks for logging per camera
        prev_set = self.previous_active_tracks.get(camera_id, set())
        for lost_id in prev_set - current_active_set:
            lost_gid = self.global_id_manager.local_to_global_map.get((camera_id, lost_id), "UNKNOWN")
            print(f"[TRACK LOST] Camera={camera_id} TrackID={lost_id} GlobalID={lost_gid}")

        self.previous_active_tracks[camera_id] = current_active_set

        # 3. Update unique observed humans per camera channel (tuple key to prevent cross-camera collisions)
        for track in active_human_tracks:
            self.unique_human_ids.add((camera_id, track.track_id))

        # 4. Compute per-frame performance metrics accurately
        t1 = time.time()
        frame_latency = max(0.001, t1 - t0)
        self.last_latency_ms = frame_latency * 1000.0
        self.last_fps = 1.0 / frame_latency

        global_summary = self.global_id_manager.get_summary_analytics()

        stats = {
            "frame_id": self.frame_count,
            "fps": round(self.last_fps, 1),
            "latency_ms": round(self.last_latency_ms, 1),
            "visible_human_count": len(active_human_tracks),
            "unique_human_count": len(self.unique_human_ids),
            "unique_global_people": global_summary.get("total_unique_global_people", global_summary.get("unique_human_count", 0)),
            "global_records": global_summary.get("records", global_summary.get("hierarchy", [])),
            "non_human_count": len(non_human_objects),
            "system_status": "ONLINE - ACTIVE TRACKING"
        }

        return active_human_tracks, non_human_objects, stats

    def reset_stats(self):
        """Reset unique counts and FPS timers for a new session."""
        self.unique_human_ids.clear()
        self.frame_count = 0
        self.start_time = time.time()
